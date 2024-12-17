<?php

require __DIR__ . '/../../vendor/autoload.php';

$queryRepo = new \mysql\QueryRepo();
$queryHandler = new \query\Handler($queryRepo);
$actorVoicesInput = new \query\ActorVoiceParams(
    keyword: "*",
    status: "*",
    sex: "*",
    rating: "*",
    age: "*",
    delivery: "*",
    page: 1
);
$actorVoices = $queryHandler->actorVoices($actorVoicesInput);
\logger\debug($actorVoices);
