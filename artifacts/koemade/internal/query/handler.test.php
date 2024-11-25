<?php

require __DIR__ . '/../../vendor/autoload.php';

$queryRepo = new \mysql\QueryRepo();
$queryHandler = new \query\Handler($queryRepo);
$actorVoices = $queryHandler->actorVoices();
\logger\debug($actorVoices);
