<?php

namespace koemade\dbadapter;

use koemade\query\search;
use koemade\util;

class SearchService implements search\Service
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
    public function actors(search\ActorsParams $params): array
    {
        $this->logger->info("Searching actors: ", $params);
        // Define items per page
        $itemsPerPage = 20;
        $page = $params->page < 1 ? 1 : $params->page;
        $offset = ($page - 1) * $itemsPerPage;

        // Start building the query
        $query = "
        SELECT
            ap.account_id AS id,
            ap.display_name AS `name`,
            ap.status AS `status`,
            ar.name AS `rank`,
            pi.path AS avatar_url
        FROM
            actor_profiles ap
        INNER JOIN
            actor_ranks ar ON ap.rank_id = ar.id
        LEFT JOIN
            profile_images pi ON ap.account_id = pi.account_id
        LEFT JOIN
            nsfw_options no ON ap.account_id = no.account_id
        WHERE 1=1
    ";

        // Apply filters based on parameters
        $filters = [
            'name_like' => !empty($params->name_like) ? " AND ap.display_name LIKE :display_name " : "",
            'status' => !empty($params->status) ? " AND ap.status = :status " : "",
            'nsfw_allowed' => $params->nsfw_allowed !== null ? " AND no.allowed = :nsfw_allowed " : "",
            'nsfw_extreme_allowed' => $params->nsfw_extreme_allowed !== null ? " AND no.extreme_allowed = :nsfw_extreme_allowed " : ""
        ];

        // Add filters to the query
        foreach ($filters as $key => $filter) {
            if (!empty($filter)) {
                $query .= $filter;
            }
        }

        // Add pagination
        $query .= " LIMIT :limit OFFSET :offset ";

        // Prepare the statement
        $stmt = $this->conn->prepare($query);

        // Bind parameters
        if (!empty($params->name_like)) {
            $stmt->bindValue(':display_name', "%{$params->name_like}%");
        }
        if (!empty($params->status)) {
            $stmt->bindValue(':status', $params->status);
        }
        if ($params->nsfw_allowed !== null) {
            $stmt->bindValue(':nsfw_allowed', $params->nsfw_allowed, \PDO::PARAM_BOOL);
        }
        if ($params->nsfw_extreme_allowed !== null) {
            $stmt->bindValue(':nsfw_extreme_allowed', $params->nsfw_extreme_allowed, \PDO::PARAM_BOOL);
        }
        $stmt->bindValue(':limit', $itemsPerPage, \PDO::PARAM_INT);
        $stmt->bindValue(':offset', $offset, \PDO::PARAM_INT);

        $this->logger->info("Query: {$stmt->queryString}");

        // Execute the query
        $stmt->execute();

        // Fetch results
        $results = $stmt->fetchAll(\PDO::FETCH_ASSOC);

        // Map results to ActorsResult objects
        $actorResults = array_map(function ($row) {
            return new search\ActorsResult(
                $row['id'],
                $row['name'],
                $row['status'],
                $row['rank'],
                $row['avatar_url'] ?? ''
            );
        }, $results);

        return $actorResults;
    }

    /**
     * @inheritDoc
     * 1. 検索条件に合うtag_idをtagsテーブルからすべて取得
     * 2. 指定されたtag_idをすべて持つvoice_idをvoice_tagテーブルから取得
     * 3. 取得したvoice_idを使ってviewから全データを取得。このviewではtagsがaggregateされている
     */
    public function voices(search\VoicesParams $query): array
    {
        $this->logger->info("Searching voices: ", $query);

        // Define items per page
        $itemsPerPage = 20;
        $page = $query->page < 1 ? 1 : $query->page;
        $offset = ($page - 1) * $itemsPerPage;

        // Base SQL for fetching data
        $sql = "SELECT * FROM voices_view WHERE 1=1";
        $params = [];

        // Add title filter
        if (!empty($query->title)) {
            $sql .= " AND voice_name LIKE :title";
            $params[':title'] = "%{$query->title}%";
        }

        // Handle tags if specified
        if (!empty($query->tags)) {
            // Step 1: Find tag_ids that match the specified tags
            $tagConditions = [];
            $tagParams = [];
            foreach ($query->tags as $index => $tag) {
                $tagConditions[] = "(category = :tag_category_$index AND name = :tag_name_$index)";
                $tagParams[":tag_category_$index"] = $tag->category;
                $tagParams[":tag_name_$index"] = $tag->name;
            }

            $tagSubquery = "
            SELECT id
            FROM tags
            WHERE " . implode(' OR ', $tagConditions) . "
        ";

            // Step 2: Find voice_ids that have ALL specified tags
            $voiceTagSubquery = "
            SELECT voice_id
            FROM voice_tag
            WHERE tag_id IN ($tagSubquery)
            GROUP BY voice_id
            HAVING COUNT(DISTINCT tag_id) = :tag_count
        ";
            $params[':tag_count'] = count($query->tags);

            // Add subquery to main SQL
            $sql .= " AND voice_id IN ($voiceTagSubquery)";

            // Merge tagParams into params
            $params = array_merge($params, $tagParams);
        }

        // Add pagination
        $sql .= " LIMIT $itemsPerPage OFFSET $offset";

        // Execute the query
        $stmt = $this->conn->prepare($sql);
        $this->logger->info("Query: {$stmt->queryString}");

        // Bind parameters
        foreach ($params as $key => $value) {
            $stmt->bindValue($key, $value);
        }

        $stmt->execute();
        $this->logger->info($stmt->queryString);

        // Process results
        $results = [];
        while ($row = $stmt->fetch(\PDO::FETCH_ASSOC)) {
            $actor = new search\VoicesResultActor();
            $actor->id = $row['actor_id'];
            $actor->name = $row['actor_name'];
            $actor->status = $row['actor_status'];
            $actor->rank = $row['actor_rank'];
            $actor->total_voices = $row['total_voices'];

            $voiceResult = new search\VoicesResult();
            $voiceResult->id = $row['voice_id'];
            $voiceResult->name = $row['voice_name'];
            $voiceResult->actor = $actor;
            $voiceResult->tags = json_decode($row['tags'], true) ?? []; // Decode JSON array
            $voiceResult->source_url = $row['source_url'];

            $results[] = $voiceResult;
        }

        return $results;
    }
}
