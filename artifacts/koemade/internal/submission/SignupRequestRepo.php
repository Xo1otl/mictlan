<?php

namespace submission;

interface SignupRequestRepo
{
    public function add(SignupRequestInput $signupRequestInput): SignupRequest;

    public function findByEmail(\common\Email $email): SignupRequest;
}
