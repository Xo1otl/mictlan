<?php

namespace actor;

class Profile
{
    public string $displayName;
    public string $category;
    public string $selfPromotion;
    public int $price;
    public ?ProfileImage $profileImage;
    public R $r;

    public function __construct(string $displayName, string $category, string $selfPromotion, int $price, R $r, ?ProfileImage $profileImage = null)
    {
        if (!in_array($category, Category::ACTOR_CATEGORIES, true)) {
            throw new \InvalidArgumentException("Invalid category: $category");
        }

        if (strlen($selfPromotion) > 200) {
            throw new \InvalidArgumentException("Self promotion must be 200 characters or less.");
        }

        $this->r = $r;
        $this->profileImage = $profileImage;
        $this->displayName = $displayName;
        $this->category = $category;
        $this->selfPromotion = $selfPromotion;
        $this->price = $price;
    }
}
