<?php

namespace actor;

class ProfileImage extends \common\IdImage
{
    public \common\Id $accountId;

    public function __construct(\common\Id $accountId, string $filename, string $mimeType, int $size, \DateTime $createdAt)
    {
        $this->accountId = $accountId;
        parent::__construct($filename, $mimeType, $size, $createdAt);
    }

}
