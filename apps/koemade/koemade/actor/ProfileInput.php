<?php

namespace koemade\actor;

class ProfileInput
{
    // actor_profiles 関連
    public ?string $displayName; // 表示名
    public ?string $selfPromotion; // 自己PR
    public ?int $price; // 価格
    public ?string $status; // ステータス

    // nsfw_options 関連
    public ?bool $nsfwAllowed; // NSFW許可
    public ?int $nsfwPrice; // NSFW価格
    public ?bool $extremeAllowed; // 過激なNSFW許可
    public ?int $extremeSurcharge; // 過激なNSFWの追加料金

    // profile_images 関連
    public ?string $profileImageMimeType; // プロフィール画像のMIMEタイプ
    public ?int $profileImageSize; // プロフィール画像のサイズ
    public ?string $profileImagePath; // プロフィール画像のパス

    public function __construct(
        string $displayName,
        string $selfPromotion = null,
        int $price = null,
        string $status = null,
        bool $nsfwAllowed = null,
        int $nsfwPrice = null,
        bool $extremeAllowed = null,
        int $extremeSurcharge = null,
        ?string $profileImageMimeType = null,
        ?int $profileImageSize = null,
        ?string $profileImagePath = null
    ) {
        $this->displayName = $displayName;
        $this->selfPromotion = $selfPromotion;
        $this->price = $price;
        $this->status = $status;
        $this->nsfwAllowed = $nsfwAllowed;
        $this->nsfwPrice = $nsfwPrice;
        $this->extremeAllowed = $extremeAllowed;
        $this->extremeSurcharge = $extremeSurcharge;
        $this->profileImageMimeType = $profileImageMimeType;
        $this->profileImageSize = $profileImageSize;
        $this->profileImagePath = $profileImagePath;
    }
}
