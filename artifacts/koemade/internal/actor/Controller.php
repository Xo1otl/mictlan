<?php

namespace actor;

class Controller
{
    private App $app;

    public function __construct(App $app)
    {
        $this->app = $app;
    }

    public function handleGetOrInitProfile(\common\Id $id): ?Profile
    {
        return $this->app->getOrInitProfile($id);
    }

    public function handleUpdateThumbnail(Storage $storage, array $fileData, \common\Id $accountId)
    {
        $profileImageFile = $fileData['profile_image'];
        if ($profileImageFile['error'] !== UPLOAD_ERR_OK) {
            \logger\fatal("upload err not ok");
        }

        $createdAt = new \DateTime();
        $profileImage = new ProfileImage($accountId, 'profile' . $accountId, $profileImageFile['type'], $profileImageFile['size'], $createdAt);
        $storage->uploadProfileImageFromTmp($profileImageFile['tmp_name'], $profileImage);

        \logger\imp($accountId);
        \logger\imp($accountId);
        \logger\imp($accountId);
        \logger\imp($accountId);
        $this->app->updateThumbnail($profileImage);
    }

    public function handleGetProfile(\common\Id $id): ?Profile
    {
        return $this->app->getProfile($id);
    }

    public function handleUpdateR(array $postData, \common\Id $accountId)
    {
        \logger\imp("updating r", $postData);
        $input = new RInput($postData['ok'], $postData['price'], $postData['hardOk'], $postData['hardSurcharge'], $accountId);
        $this->app->editR($input);
    }

    public function handleUpdateProfile(array $postData, \common\Id $accountId)
    {
        \logger\imp($postData, $accountId);
        \logger\imp($postData, $accountId);
        \logger\imp($postData, $accountId);
        \logger\imp($postData, $accountId);
        $displayName = $postData['displayName'] ?? false;
        $category = $postData['category'] ?? 0;
        $selfPromotion = $postData['selfPromotion'] ?? false;
        $price = $postData['price'] ?? 0;

        $input = new ProfileInput($displayName, $category, $selfPromotion, $price, $accountId);
        $this->app->editProfile($input);
    }
}
