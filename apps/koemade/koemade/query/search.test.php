<?php

require_once __DIR__ . '/../kernel/bootstrap.php';

use koemade\dbadapter;
use koemade\query\search;
use koemade\query\valueObjects;

$searchService = new dbadapter\SearchService();

$actorsParams = new search\ActorsParams("", "受付中", 1);
$actors = $searchService->actors($actorsParams);

$tag = new valueObjects\Tag("mood", "犯罪者");
$voicesParams = new search\VoicesParams("フェミニスト", [$tag], 1);
$voices = $searchService->voices($voicesParams);

$logger->info("Actors: ", $actors);
$logger->info("Voices: ", $voices);
