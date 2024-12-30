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
    $status = $input->status;
    $sex = $input->sex;
    $tag = $input->tag;
    $age = $input->age;
  }

  function actors($input): array
  {
    // Assuming you need to fetch actors from the database and create Actor objects
    $query = "SELECT id, name, status, rank, total_voices FROM actors"; // Adjust the query as needed
    $result = $this->mysqli->query($query);

    $actors = [];
    while ($row = $result->fetch_assoc()) {
      $actors[] = new \query\Actor(
        $row['id'],
        $row['name'],
        $row['status'],
        $row['rank'],
        $row['total_voices']
      );
    }

    return $actors;
  }
  function actorProfile($input): \query\ActorProfile
  {
    return new \query\ActorProfile();
  }
  /**
   * @inheritDoc
   */
  public function voice(string $voiceId): \query\VoiceDetail
  {
  }
}
