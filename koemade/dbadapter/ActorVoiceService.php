<?php

namespace koemade\dbadapter;

use koemade\actor;
use koemade\util;

class ActorVoiceService implements actor\VoiceService
{
    private \PDO $conn;
    private util\Logger $logger;

    public function __construct()
    {
        $this->conn = DBConnection::getInstance();
        $this->logger = util\Logger::getInstance();
    }


    /**
     * @inheritDoc
     */
    public function editTitle(string $voice_id, string $newTitle): void
    {
        try {
            $stmt = $this->conn->prepare("UPDATE voices SET title = :newTitle WHERE id = :voice_id");
            $stmt->bindParam(':newTitle', $newTitle, \PDO::PARAM_STR);
            $stmt->bindParam(':voice_id', $voice_id, \PDO::PARAM_INT);
            $stmt->execute();

            $this->logger->info("Title updated for voice_id: $voice_id");
        } catch (\PDOException $e) {
            $this->logger->error("Failed to update title for voice_id: $voice_id - " . $e->getMessage());
            throw new \RuntimeException("Failed to update title", 0, $e);
        }
    }

    /**
     * @inheritDoc
     */
    public function updateTags(string $voice_id, array $tagIds): void
    {
        try {
            // トランザクションを開始
            $this->conn->beginTransaction();

            // 既存のタグを削除
            $stmt = $this->conn->prepare("DELETE FROM voice_tag WHERE voice_id = :voice_id");
            $stmt->bindParam(':voice_id', $voice_id, \PDO::PARAM_INT);
            $stmt->execute();

            // 新しいタグを追加
            $stmt = $this->conn->prepare("INSERT INTO voice_tag (voice_id, tag_id) VALUES (:voice_id, :tag_id)");
            foreach ($tagIds as $tagId) {
                $stmt->bindParam(':voice_id', $voice_id, \PDO::PARAM_INT);
                $stmt->bindParam(':tag_id', $tagId, \PDO::PARAM_INT);
                $stmt->execute();
            }

            // トランザクションをコミット
            $this->conn->commit();

            $this->logger->info("Tags updated for voice_id: $voice_id");
        } catch (\PDOException $e) {
            // エラーが発生した場合はロールバック
            $this->conn->rollBack();
            $this->logger->error("Failed to update tags for voice_id: $voice_id - " . $e->getMessage());
            throw new \RuntimeException("Failed to update tags", 0, $e);
        }
    }
}
