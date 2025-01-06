<?php

namespace koemade\dbadapter;

use koemade\query;
use koemade\util;

class QueryRepository implements query\repository\Service
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
    public function actorFeed(string $actor_id): query\repository\ActorFeed
    {
        $stmt = $this->conn->prepare("
            SELECT * FROM actor_feed_view WHERE actor_id = :actor_id
        ");
        $stmt->execute(['actor_id' => $actor_id]);
        $rows = $stmt->fetchAll(\PDO::FETCH_ASSOC);

        if (empty($rows)) {
            throw new \RuntimeException("Actor not found with ID: $actor_id");
        }

        $actor = new query\repository\Actor();
        $actor->id = $rows[0]['actor_id'];
        $actor->name = $rows[0]['actor_name'];
        $actor->status = $rows[0]['actor_status'];
        $actor->rank = $rows[0]['actor_rank'];
        $actor->description = $rows[0]['actor_description'];
        $actor->avator_url = $rows[0]['actor_avatar_url'];
        $actor->price = new query\valueObjects\Price(
            $rows[0]['actor_price_default'],
            $rows[0]['actor_price_nsfw'],
            $rows[0]['actor_price_nsfw_extreme'],
        );

        $sampleVoices = [];
        $voiceMap = [];

        foreach ($rows as $row) {
            if (!isset($voiceMap[$row['voice_id']])) {
                $voice = new query\repository\SampleVoice();
                $voice->id = $row['voice_id'];
                $voice->name = $row['voice_title'];
                $voice->source_url = $row['voice_source_url'];
                $voice->tags = [];

                $voiceMap[$row['voice_id']] = $voice;
                $sampleVoices[] = $voice;
            }

            if ($row['tag_id']) {
                $tag = new query\valueObjects\Tag($row['tag_name'], $row['tag_category']);
                $voiceMap[$row['voice_id']]->tags[] = $tag;
            }
        }

        $actorFeed = new query\repository\ActorFeed();
        $actorFeed->actor = $actor;
        $actorFeed->sampleVoices = $sampleVoices;

        return $actorFeed;
    }

    /**
     * @inheritDoc
     */
    public function findVoiceWithTagsByID(string $voice_id): query\repository\VoiceWithTags
    {
        // ボイス情報とアカウント情報、タグ情報を取得するクエリ
        $stmt = $this->conn->prepare("
            SELECT
                v.id AS voice_id,
                v.title AS voice_title,
                v.path AS voice_filename,
                v.created_at AS voice_created_at,
                a.id AS account_id,
                a.username AS account_username,
                pi.path AS account_avatar_url,
                t.id AS tag_id,
                t.name AS tag_name,
                t.category AS tag_category
            FROM
                voices v
            JOIN
                accounts a ON v.account_id = a.id
            LEFT JOIN
                profile_images pi ON a.id = pi.account_id
            LEFT JOIN
                voice_tag vt ON v.id = vt.voice_id
            LEFT JOIN
                tags t ON vt.tag_id = t.id
            WHERE
                v.id = :voice_id
        ");
        $stmt->execute(['voice_id' => $voice_id]);
        $rows = $stmt->fetchAll(\PDO::FETCH_ASSOC);

        if (empty($rows)) {
            throw new \RuntimeException("Voice not found with ID: $voice_id");
        }

        // VoiceWithTags オブジェクトを構築
        $voiceWithTags = new query\repository\VoiceWithTags();
        $voiceWithTags->id = $rows[0]['voice_id'];
        $voiceWithTags->title = $rows[0]['voice_title'];
        $voiceWithTags->filename = $rows[0]['voice_filename'];
        $voiceWithTags->created_at = $rows[0]['voice_created_at'];

        // アカウント情報を設定
        $voiceWithTags->account = [
            'id' => $rows[0]['account_id'],
            'username' => $rows[0]['account_username'],
            'avator_url' => $rows[0]['account_avatar_url'] ?? '', // アバターURLがない場合は空文字
        ];

        // タグ情報を設定
        $voiceWithTags->tags = [];
        foreach ($rows as $row) {
            if ($row['tag_id']) {
                $tag = new query\valueObjects\Tag($row['tag_name'], $row['tag_category']);
                $voiceWithTags->tags[] = $tag;
            }
        }

        return $voiceWithTags;
    }
}
