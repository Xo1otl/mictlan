<?php

namespace submission;

/**
 * submissionできるapp
 *
 * アカウント作成申込みを提出できる
 */
class App
{
    private SignupRequestRepo $signupRequestRepo;
    private NotificationService $notificationService;

    public function __construct(SignupRequestRepo $repo, NotificationService $notificationService)
    {
        $this->signupRequestRepo = $repo;
        $this->notificationService = $notificationService;
    }

    public function submitSignupRequest(SignupRequestInput $signupRequestInput)
    {
        $signupRequest = $this->signupRequestRepo->add($signupRequestInput);
        $this->notificationService->notify($signupRequest);
    }
}
