<?php

require_once __DIR__ . '/../kernel/bootstrap.php';

use koemade\dbadapter;
use koemade\query\search;
use koemade\query\valueObjects;

$searchService = new dbadapter\SearchService();

$actorsParams = new search\ActorsParams(name_like: "", status: "受付中", page: 1);
$actors = $searchService->actors($actorsParams);

$tag = new valueObjects\Tag(category: "mood", name: "犯罪者");
$voicesParams = new search\VoicesParams(title: "フェミニスト", tags: [$tag], page: 1);
$voices = $searchService->voices($voicesParams);

$logger->info("Actors: ", $actors);
$logger->info("Voices: ", $voices);
