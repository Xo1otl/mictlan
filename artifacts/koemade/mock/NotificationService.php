<?php

namespace mock;

class NotificationService implements \submission\NotificationService
{

    public function notify(\submission\SignupRequest $signupRequest): void
    {
        echo "MockNotificationService";
        var_dump($signupRequest);
    }
}
