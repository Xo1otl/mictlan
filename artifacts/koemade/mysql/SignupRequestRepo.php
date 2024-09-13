<?php

namespace mysql;

class SignupRequestRepo implements \submission\SignupRequestRepo
{
    private \mysqli $mysqli;

    public function __construct()
    {
        $this->mysqli = DbConnection::getConnection();
    }

    public function add(\submission\SignupRequestInput $signupRequestInput): \submission\SignupRequest
    {
        $stmt = $this->mysqli->prepare("INSERT INTO signup_requests (
            name, furigana, address, email, tel, bank_name, branch_name, account_number, self_promotion
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");

        $stmt->bind_param(
            "sssssssss",
            $signupRequestInput->japaneseName->name,
            $signupRequestInput->japaneseName->furigana,
            $signupRequestInput->address->text,
            $signupRequestInput->email->text,
            $signupRequestInput->tel->number,
            $signupRequestInput->beneficiaryInfo->bankName,
            $signupRequestInput->beneficiaryInfo->branchName,
            $signupRequestInput->beneficiaryInfo->accountNumber,
            $signupRequestInput->selfPromotion,
        );

        if (!$stmt->execute()) {
            \logger\err("Failed to save SignupRequest: " . $stmt->error);
        }

        $id = new \common\Id($stmt->insert_id);
        $stmt->close();
        return new \submission\SignupRequest(
            $id,
            $signupRequestInput->email,
            $signupRequestInput->japaneseName,
            $signupRequestInput->address,
            $signupRequestInput->tel,
            $signupRequestInput->idImage,
            $signupRequestInput->beneficiaryInfo,
            $signupRequestInput->selfPromotion
        );
    }

    public function findByEmail(\common\Email $email): \submission\SignupRequest
    {
        \logger\fatal("not implemented");
    }
}
