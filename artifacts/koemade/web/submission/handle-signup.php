<?php

ignore_user_abort(true);

require __DIR__ . '/../../vendor/autoload.php';

$signupRequestRepo = new \mysql\SignupRequestRepo();
$notificationService = new \mail\NotificationService();
$submissionApp = new \submission\App($signupRequestRepo, $notificationService);
$submissionController = new \submission\Controller($submissionApp);
$storage = new \filesystem\SubmissionStorage();
$submissionController->submitSignupRequest($_POST, $_FILES, $storage);
