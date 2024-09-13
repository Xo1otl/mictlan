<?php

namespace mysql;

require __DIR__ . '/../vendor/autoload.php';

use function utils\generateUUID;

function TestVoiceRepoAdd(string $title)
{
    $db = new VoiceRepo();
    $id = new \common\Id(1);
    $input = new \voice\Input($id, $title, $title, "audio/mpeg", 1024 * 5, new \DateTime());
    $db->upload($input);
}

function TestVoiceRepoGetAll()
{
    $db = new VoiceRepo();
    $id = new \common\Id(1);
    $voices = $db->getAll($id);
    \logger\debug($voices);
}

function TestEditTag(string $title)
{
    $db = new VoiceRepo();
    $input = new \voice\EditTagInput(new \common\Id(1), $title, "10代", '大人しい');
    $db->editTag($input);
}

$title = generateUUID();
TestVoiceRepoAdd($title);
TestEditTag($title);
