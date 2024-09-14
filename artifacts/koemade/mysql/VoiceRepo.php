<?php

namespace mysql;

use voice\EditTagInput;
use voice\Tag;

class VoiceRepo implements \voice\Repo
{
    private \mysqli $mysqli;

    public function __construct()
    {
        $this->mysqli = DbConnection::getConnection();
    }

    function upload(\voice\Input $input)
    {
        $stmt = $this->mysqli->prepare('
        INSERT INTO voices (title, account_id, mime_type, filename, created_at)
        VALUES (?, ?, ?, ?, ?)
        ');

        if (!$stmt) {
            throw new \RuntimeException("Failed to prepare statement: " . $this->mysqli->error);
        }

        $title = $input->title;
        $accountId = $input->accountId->value;
        $mimeType = $input->mimeType;
        $filename = $input->filename;
        $createdAt = $input->createdAt->format('Y-m-d H:i:s');
        $stmt->bind_param('sisss', $title, $accountId, $mimeType, $filename, $createdAt);

        if (!$stmt->execute()) {
            throw new \RuntimeException("Failed to execute statement: " . $stmt->error);
        }

        $stmt->close();
    }

    /**
     * @param \common\Id $accountId
     * @return array|\voice\Voice[]
     */
    function getAll(\common\Id $accountId): array
    {
        $stmt = $this->mysqli->prepare('
        SELECT 
            v.id AS voice_id,
            v.title AS voice_title,
            v.account_id,
            v.mime_type,
            v.filename,
            v.created_at,
            vt.id AS tag_id,
            vt.tag_name,
            vt.tag_category
        FROM 
            voices v
        LEFT JOIN 
            voice_tag_map vtm ON v.id = vtm.voice_id
        LEFT JOIN 
            voice_tags vt ON vtm.tag_id = vt.id
        WHERE 
            v.account_id = ?;
        ');

        if (!$stmt) {
            throw new \RuntimeException("Failed to prepare statement: " . $this->mysqli->error);
        }

        $accountIdValue = $accountId->value;
        $stmt->bind_param('i', $accountIdValue);

        if (!$stmt->execute()) {
            throw new \RuntimeException("Failed to execute statement: " . $stmt->error);
        }

        $result = $stmt->get_result();
        $voices = [];

        while ($row = $result->fetch_assoc()) {
            $tag = null;
            if ($row['tag_name'] !== null && $row['tag_category'] !== null) {
                $tag = new Tag($row['tag_category'], $row['tag_name']);
            }
            if (!isset($voices[$row['voice_id']])) {
                \logger\debug($row);
                $voices[$row['voice_id']] = new \voice\Voice(
                    new \common\Id($row['voice_id']),
                    new \common\Id($row['account_id']),
                    $row['filename'],
                    $row['voice_title'],
                    $row['mime_type'],
                    \DateTime::createFromFormat('Y-m-d H:i:s', $row['created_at']),
                    $tag ? [$tag] : []
                );
            } else {
                if ($tag !== null) {
                    $voices[$row['voice_id']]->tags[] = $tag;
                }
            }
        }

        $stmt->close();

        return $voices;
    }

    function editTag(EditTagInput $input)
    {
        $this->mysqli->begin_transaction();
        try {
            // Get voice ID
            $stmt = $this->mysqli->prepare("SELECT id FROM voices WHERE id = ?");
            $stmt->bind_param("i", $input->voiceId->value);
            $stmt->execute();
            $stmt->bind_result($voiceId);
            $stmt->fetch();
            $stmt->close();

            if (!$voiceId) {
                throw new \Exception("Voice not found.");
            }

            // Get tag IDs
            $stmt = $this->mysqli->prepare("SELECT id FROM voice_tags WHERE tag_name = ?");
            $stmt->bind_param("s", $input->ageTag);
            $stmt->execute();
            $stmt->bind_result($ageTagId);
            $stmt->fetch();
            $stmt->free_result();

            $stmt->bind_param("s", $input->characterTag);
            $stmt->execute();
            $stmt->bind_result($characterTagId);
            $stmt->fetch();
            $stmt->close();

            if (!$ageTagId || !$characterTagId) {
                throw new \Exception("Invalid tag IDs fetched.");
            }

            // TODO: このSQLは簡単にできるので修正する
            // Remove existing tags of the same category
            $stmt = $this->mysqli->prepare("
            DELETE FROM voice_tag_map 
            WHERE voice_id = ? AND tag_id IN 
            (SELECT id FROM voice_tags WHERE tag_category IN 
            ((SELECT tag_category from voice_tags where id = ?), 
            (SELECT tag_category from voice_tags where id = ?)))
            ");
            $stmt->bind_param("iss", $voiceId, $ageTagId, $characterTagId);
            $stmt->execute();
            $stmt->close();

            // Insert new tags
            $stmt = $this->mysqli->prepare("INSERT INTO voice_tag_map (voice_id, tag_id) VALUES (?, ?), (?, ?)");
            $stmt->bind_param("iiii", $voiceId, $ageTagId, $voiceId, $characterTagId);
            $stmt->execute();
            $stmt->close();

            // update voice title
            $stmt = $this->mysqli->prepare("UPDATE voices SET title = ? WHERE id = ?");
            $stmt->bind_param("si", $input->title, $voiceId);
            $stmt->execute();
            $stmt->close();

            // Commit transaction
            $this->mysqli->commit();
        } catch (\Exception $e) {
            $this->mysqli->rollback();
            \logger\fatal("Transaction failed: " . $e->getMessage());
        }
    }

    function deleteById(\common\Id $voiceId, \common\Id $accountId)
    {
        \logger\imp("delete account ", $voiceId, $accountId);
        $stmt = $this->mysqli->prepare("DELETE FROM voices WHERE id = ? AND account_id = ?");
        $stmt->bind_param('ii', $voiceId->value, $accountId->value);
        $stmt->execute();
        $stmt->close();
    }
}
