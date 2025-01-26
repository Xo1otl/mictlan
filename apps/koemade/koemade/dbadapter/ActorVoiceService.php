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
    public function newVoice(actor\NewVoiceInput $input): void
    {
        try {
            // トランザクションを開始
            $this->conn->beginTransaction();

            // 新しいボイスを追加
            $stmt = $this->conn->prepare("INSERT INTO voices (account_id, title, path, mime_type) VALUES (:account_id, :title, :path, :mime_type)");
            $stmt->bindParam(':account_id', $input->sub, \PDO::PARAM_STR);
            $stmt->bindParam(':title', $input->title, \PDO::PARAM_STR);
            $stmt->bindParam(':path', $input->path, \PDO::PARAM_STR);
            $stmt->bindParam(':mime_type', $input->mime_type, \PDO::PARAM_STR);
            $stmt->execute();

            // 新しく追加されたボイスのIDを取得
            $voice_id = $this->conn->lastInsertId();

            // タグが指定されている場合、追加する
            if ($input->tagIds !== null) {
                $stmt = $this->conn->prepare("INSERT INTO voice_tag (voice_id, tag_id) VALUES (:voice_id, :tag_id)");
                foreach ($input->tagIds as $tagId) {
                    $stmt->bindParam(':voice_id', $voice_id, \PDO::PARAM_INT);
                    $stmt->bindParam(':tag_id', $tagId, \PDO::PARAM_INT);
                    $stmt->execute();
                }
                $this->logger->info("Tags added for new voice_id: $voice_id");
            }

            // トランザクションをコミット
            $this->conn->commit();

            $this->logger->info("New voice created with title: $input->title, voice_id: $voice_id");
        } catch (\PDOException $e) {
            // エラーが発生した場合はロールバック
            $this->conn->rollBack();
            $this->logger->error("Failed to create new voice - " . $e->getMessage());
            throw new \RuntimeException("Failed to create new voice", 0, $e);
        }
    }

    /**
     * @inheritDoc
     */
    public function updateVoice(actor\UpdateVoiceInput $input): void
    {
        try {
            // トランザクションを開始
            $this->conn->beginTransaction();

            // 指定された voice_id が sub に属しているかを確認
            $stmt = $this->conn->prepare("SELECT id FROM voices WHERE id = :voice_id AND account_id = :account_id");
            $stmt->bindParam(':voice_id', $input->voice_id, \PDO::PARAM_INT);
            $stmt->bindParam(':account_id', $input->sub, \PDO::PARAM_INT);
            $stmt->execute();

            if ($stmt->fetch() === false) {
                throw new \RuntimeException("Voice not found or does not belong to the account");
            }

            // タイトルが指定されている場合、更新する
            if ($input->newTitle !== null) {
                $stmt = $this->conn->prepare("UPDATE voices SET title = :newTitle WHERE id = :voice_id");
                $stmt->bindParam(':newTitle', $input->newTitle, \PDO::PARAM_STR);
                $stmt->bindParam(':voice_id', $input->voice_id, \PDO::PARAM_INT);
                $stmt->execute();
                $this->logger->info("Title updated for voice_id: $input->voice_id");
            }

            // タグが指定されている場合、更新する
            if ($input->tagIds !== null) {
                // 既存のタグを削除
                $stmt = $this->conn->prepare("DELETE FROM voice_tag WHERE voice_id = :voice_id");
                $stmt->bindParam(':voice_id', $input->voice_id, \PDO::PARAM_INT);
                $stmt->execute();

                // 新しいタグを追加
                $stmt = $this->conn->prepare("INSERT INTO voice_tag (voice_id, tag_id) VALUES (:voice_id, :tag_id)");
                foreach ($input->tagIds as $tagId) {
                    $stmt->bindParam(':voice_id', $input->voice_id, \PDO::PARAM_INT);
                    $stmt->bindParam(':tag_id', $tagId, \PDO::PARAM_INT);
                    $stmt->execute();
                }
                $this->logger->info("Tags updated for voice_id: $input->voice_id");
            }

            // トランザクションをコミット
            $this->conn->commit();
        } catch (\PDOException $e) {
            // エラーが発生した場合はロールバック
            $this->conn->rollBack();
            $this->logger->error("Failed to update voice for voice_id: $input->voice_id - " . $e->getMessage());
            throw new \RuntimeException("Failed to update voice", 0, $e);
        }
    }

    /**
     * ボイスを削除する
     *
     * @param string $sub アカウントID
     * @param string $voice_id 削除するボイスのID
     * @throws \RuntimeException 削除に失敗した場合
     */
    public function deleteVoice(string $sub, string $voice_id): void
    {
        try {
            // トランザクションを開始
            $this->conn->beginTransaction();

            // 指定された voice_id が sub に属しているかを確認
            $stmt = $this->conn->prepare("SELECT id FROM voices WHERE id = :voice_id AND account_id = :account_id");
            $stmt->bindParam(':voice_id', $voice_id, \PDO::PARAM_INT);
            $stmt->bindParam(':account_id', $sub, \PDO::PARAM_STR);
            $stmt->execute();

            if ($stmt->fetch() === false) {
                throw new \RuntimeException("Voice not found or does not belong to the account");
            }

            // 関連するタグを削除
            $stmt = $this->conn->prepare("DELETE FROM voice_tag WHERE voice_id = :voice_id");
            $stmt->bindParam(':voice_id', $voice_id, \PDO::PARAM_INT);
            $stmt->execute();

            // ボイスを削除
            $stmt = $this->conn->prepare("DELETE FROM voices WHERE id = :voice_id");
            $stmt->bindParam(':voice_id', $voice_id, \PDO::PARAM_INT);
            $stmt->execute();

            // トランザクションをコミット
            $this->conn->commit();

            $this->logger->info("Voice deleted for voice_id: $voice_id");
        } catch (\PDOException $e) {
            // エラーが発生した場合はロールバック
            $this->conn->rollBack();
            $this->logger->error("Failed to delete voice for voice_id: $voice_id - " . $e->getMessage());
            throw new \RuntimeException("Failed to delete voice", 0, $e);
        }
    }
}