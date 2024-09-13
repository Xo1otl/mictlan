<?php

namespace submission;

interface NotificationService
{
    public function notify(SignupRequest $signupRequest): void;
}
