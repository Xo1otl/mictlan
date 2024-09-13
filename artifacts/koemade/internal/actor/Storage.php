<?php

namespace actor;

interface Storage
{
    function uploadProfileImageFromTmp(string $tmpPath, ProfileImage $profileImage);
}
