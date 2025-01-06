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
     */
    public function voices(search\VoicesParams $query): array
    {
        $this->logger->info("Searching voices: ", $query);

        // Define items per page
        $itemsPerPage = 20;
        $page = $query->page < 1 ? 1 : $query->page;
        $offset = ($page - 1) * $itemsPerPage;

        $sql = "SELECT * FROM voices_view WHERE voice_name LIKE :title"; // Changed 'title' to 'voice_name'
        $params = [':title' => "%{$query->title}%"];

        if (!empty($query->tags)) {
            $tagConditions = [];
            foreach ($query->tags as $index => $tag) {
                $tagConditions[] = "(tag_category = :tag_category_$index AND tag_name = :tag_name_$index)";
                $params[":tag_category_$index"] = $tag->category;
                $params[":tag_name_$index"] = $tag->name;
            }
            $sql .= " AND (" . implode(' OR ', $tagConditions) . ")";
        }

        // Directly insert itemsPerPage and offset into the SQL
        $sql .= " LIMIT $itemsPerPage OFFSET $offset";

        $stmt = $this->conn->prepare($sql);
        $this->logger->info("Query: {$stmt->queryString}");

        // Bind parameters excluding offset since it's directly in the SQL
        foreach ($params as $key => $value) {
            $stmt->bindValue($key, $value);
        }

        $stmt->execute();

        $results = [];
        while ($row = $stmt->fetch(\PDO::FETCH_ASSOC)) {
            $actor = new search\VoicesResultActor();
            $actor->id = $row['actor_id'];
            $actor->name = $row['actor_name'];
            $actor->status = $row['actor_status'];
            $actor->rank = $row['actor_rank'];
            $actor->total_voices = $row['total_voices'];

            $result = new search\VoicesResult();
            $result->id = $row['voice_id'];
            $result->name = $row['voice_name'];
            $result->actor = $actor;
            $result->tags = [];
            $result->source_url = $row['source_url'];

            if ($row['tag_id']) {
                $result->tags[] = [
                    'category' => $row['tag_category'],
                    'name' => $row['tag_name']
                ];
            }

            $results[] = $result;
        }

        return $results;
    }
}
