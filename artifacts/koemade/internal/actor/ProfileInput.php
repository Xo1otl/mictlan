<?php

namespace actor;

class ProfileInput
{
    public string $displayName;
    public string $category;
    public string $selfPromotion;
    public string $price;
    public \common\Id $accountId;

    public function __construct(string $displayName, string $category, string $selfPromotion, int $price, \common\Id $accountId)
    {
        if (!in_array($category, Category::ACTOR_CATEGORIES, true)) {
            throw new \InvalidArgumentException("Invalid category: $category");
        }

        if (strlen($selfPromotion) > 200) {
            throw new \InvalidArgumentException("Self promotion must be 200 characters or less.");
        }

        $this->displayName = $displayName;
        $this->category = $category;
        $this->selfPromotion = $selfPromotion;
        $this->price = $price;
        $this->accountId = $accountId;
    }
}
