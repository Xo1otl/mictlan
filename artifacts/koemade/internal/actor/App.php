<?php

namespace actor;

class App
{
    private ProfileRepo $profileRepo;

    public function __construct(ProfileRepo $profileRepo)
    {
        $this->profileRepo = $profileRepo;
    }

    public function editProfile(ProfileInput $input)
    {
        $this->profileRepo->addOrEdit($input);
    }

    public function editR(RInput $input)
    {
        $this->profileRepo->addOrEditR($input);
    }

    public function updateThumbnail(ProfileImage $profileImage)
    {
        $this->profileRepo->updateThumbnail($profileImage);
    }

    public function getProfile(\common\Id $accountId): Profile
    {
        return $this->profileRepo->findById($accountId);
    }

    public function getOrInitProfile(\common\Id $accountId): Profile
    {
        try {
            return $this->getProfile($accountId);
        } catch (\Exception $e) {

            \logger\imp("initialize profile because the profile was not complete");
            try {
                $r = $this->profileRepo->findR($accountId);
                $profile = new Profile("", Category::AMATEUR, "", 0, $r);
            } catch (\Exception $e) {
                $r = new R(0, false, 0, false);
                $profile = new Profile("", Category::AMATEUR, "", 0, $r);
            }

            try {
                $profileImage = $this->profileRepo->findProfileImage($accountId);
                $profile->profileImage = $profileImage;
            } catch (\Exception $e) {
                \logger\info("profile image not found");
            }

            return $profile;
        }
    }
}
