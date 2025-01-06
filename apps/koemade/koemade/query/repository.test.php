<?php

require_once __DIR__ . '/../kernel/bootstrap.php';

use koemade\dbadapter;

$repository = new dbadapter\QueryRepository();
$feed = $repository->actorFeed(3);
$voiceWithTags = $repository->findVoiceWithTagsByID(18);
// $logger->info("Actor Feed: ", (array)$feed);
$logger->info("Voice with tags", (array) $voiceWithTags);
