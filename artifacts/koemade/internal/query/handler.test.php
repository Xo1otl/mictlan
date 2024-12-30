<?php

require __DIR__ . '/../../vendor/autoload.php';

$queryRepo = new \mysql\QueryRepo();
$queryHandler = new \query\Handler($queryRepo);
$actorVoicesInput = new \query\ActorVoiceParams(
    keyword: "keyword",
    status: "status",
    sex: "sex",
    tag: "name",
    age: "age",
    delivery: "delivery",
    page: 1
);
$actorVoices = $queryHandler->actorVoices($actorVoicesInput);
\logger\debug($actorVoices);
