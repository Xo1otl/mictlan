<?php

namespace actor;

interface ProfileRepo
{
    public function updateThumbnail(ProfileImage $profileImage);

    public function addOrEdit(ProfileInput $input);

    public function addOrEditR(NSFWInput $input);

    public function findById(\common\Id $accountId): Profile;

    public function findR(\common\Id $accountId): NSFWOptions;

    public function findProfileImage(\common\Id $accountId): ProfileImage;
}
