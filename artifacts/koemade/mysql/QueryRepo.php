<?php

namespace mysql;

class QueryRepo implements \query\Repo
{
  private \mysqli $mysqli;

  public function __construct()
  {
    $this->mysqli = DbConnection::getConnection();
  }
  function actorVoices($input): array
  {
    $keyword = $input->keyword;
    $page = $input->page;
    $perPage = 10; // You can adjust this value
    $offset = ($page - 1) * $perPage;

    $query = "SELECT 
                    v.id AS voice_id,
                    v.title AS voice_title,
                    v.filename AS voice_filename,
                    a.id AS actor_id,
                    p.display_name AS actor_name,
                    GROUP_CONCAT(vt.tag_name) AS tags
                  FROM 
                    voices v
                  JOIN 
                    accounts a ON v.account_id = a.id
                  JOIN 
                    profiles p ON a.id = p.account_id
                  LEFT JOIN 
                    voice_tag_map vtm ON v.id = vtm.voice_id
                  LEFT JOIN 
                    voice_tags vt ON vtm.tag_id = vt.id
                  WHERE 
                    v.title LIKE ?
                  GROUP BY 
                    v.id, a.id, p.display_name
                  LIMIT ?, ?";

    $stmt = $this->mysqli->prepare($query);
    $likeKeyword = "%" . $keyword . "%";
    $stmt->bind_param("sii", $likeKeyword, $offset, $perPage);
    $stmt->execute();
    $result = $stmt->get_result();

    $actorVoices = [];
    while ($row = $result->fetch_assoc()) {
      $tags = $row['tags'] ? explode(',', $row['tags']) : [];

      // Placeholder for ratings and actor details - you'll need to adjust these based on your actual data structure
      $ratings = [
        'overall' => 0.0, // Replace with actual rating if available
        'clarity' => 0.0, // Replace with actual rating if available
        'naturalness' => 0.0, // Replace with actual rating if available
      ];
      $actor = [
        'id' => $row['actor_id'],
        'name' => $row['actor_name'],
        'status' => 'unknown', // Replace with actual status if available
        'rank' => 'unknown', // Replace with actual rank if available
        'total_voices' => 0, // Replace with actual count if available
      ];

      $actorVoices[] = new \query\ActorVoice(
        $row['voice_id'],
        $row['voice_title'],
        $row['voice_filename'], // You might want to construct a full URL here
        $tags,
        $ratings,
        $actor
      );
    }

    return $actorVoices;
  }
  function actors($input): array
  {
    return new \query\Actor();
  }
  function actorProfile($input): \query\ActorProfile
  {
    return new \query\ActorProfile();
  }
}
