<?php

namespace mock;

class SignupRequestRepo implements \submission\SignupRequestRepo
{
    public function add(\submission\SignupRequestInput $signupRequestInput): \submission\SignupRequest
    {
        echo "MockSignupRequestRepo:";
        var_dump($signupRequestInput);
        \logger\fatal("not implemented");
    }

    public function findByEmail(\common\Email $email): \submission\SignupRequest
    {
        \logger\fatal("not implemented");
    }
}
