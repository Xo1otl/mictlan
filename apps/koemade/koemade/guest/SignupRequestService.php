<?php

namespace koemade\guest;

interface SignupRequestService
{
    public function notify(SignupRequest $signupRequest): void;
}
