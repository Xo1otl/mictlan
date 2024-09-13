<?php

namespace voice;

interface Storage
{
    function uploadVoice(string $tmpPath, Input $input);
}
