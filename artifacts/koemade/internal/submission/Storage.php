<?php

namespace submission;

interface Storage
{
    public function uploadIdImageFromTmp(string $tmpPath, \common\IdImage $idImage);
}
