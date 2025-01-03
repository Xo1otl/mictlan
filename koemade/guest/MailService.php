<?php

namespace koemade\guest;

use koemade\util;

class MailService implements SignupRequestService
{
    private const MAIL_CONFIG = [
        'ENCODING' => '7bit',
        'HOST' => 's16.valueserver.jp',
        'FROM' => 'demo3@koemade.net',
        'FROM_NAME' => 'コエメイド運営',
        'PHP_LANGUAGE' => 'japanese',
        'PHP_INTERNAL_ENCODING' => 'UTF-8',
        'CHARSET' => 'iso-2022-jp',
        'PORT' => 25,
        'SMTP_SECURE' => \PHPMailer\PHPMailer\PHPMailer::ENCRYPTION_STARTTLS,
    ];

    private util\Logger $logger;

    public function __construct()
    {
        $this->logger = util\Logger::getInstance();
    }

    public function notify(SignupRequest $signupRequest): void
    {
        $this->logger->info('メール送信を開始します。');

        $mail = new \PHPMailer\PHPMailer\PHPMailer(true);
        $mail->SMTPDebug = 4;

        try {
            mb_language(self::MAIL_CONFIG['PHP_LANGUAGE']);
            mb_internal_encoding(self::MAIL_CONFIG['PHP_INTERNAL_ENCODING']);

            $this->sendMail($mail, $signupRequest);
            $this->logger->info('メール送信に成功しました。');
        } catch (\PHPMailer\PHPMailer\Exception $e) {
            $this->logger->error('メール送信に失敗しました。', ['error' => $e->getMessage()]);
        }
    }

    private function sendMail(\PHPMailer\PHPMailer\PHPMailer $mail, SignupRequest $signupRequest): void
    {
        $mail->CharSet = self::MAIL_CONFIG['CHARSET'];
        $mail->Encoding = self::MAIL_CONFIG['ENCODING'];
        $mail->isSMTP();
        $mail->Host = self::MAIL_CONFIG['HOST'];
        $mail->SMTPAuth = false;
        $mail->SMTPSecure = self::MAIL_CONFIG['SMTP_SECURE'];
        $mail->Port = self::MAIL_CONFIG['PORT'];

        $mail->setFrom(self::MAIL_CONFIG['FROM'], mb_encode_mimeheader(self::MAIL_CONFIG['FROM_NAME']));
        $mail->addAddress(self::MAIL_CONFIG['FROM']);
        $mail->addAttachment($signupRequest->idImagePath);
        $mail->isHTML(false);
        $mail->Subject = mb_encode_mimeheader('ユーザー登録申請の通知');
        $mail->Body = $this->constructBody($signupRequest);

        $mail->send();
    }

    private function constructBody(SignupRequest $signupRequest): string
    {
        $template = <<<EOD
ユーザーからアカウント登録の申請がありました。

メールアドレス: %s
名前: %s
ふりがな: %s
住所: %s
電話番号: %s
自己PR: %s
銀行名: %s
支店名: %s
口座番号: %s

ID画像:
EOD;

        return sprintf(
            $template,
            $signupRequest->email,
            $signupRequest->japaneseName->name,
            $signupRequest->japaneseName->furigana,
            $signupRequest->address,
            $signupRequest->tel,
            $signupRequest->selfPromotion,
            $signupRequest->beneficiaryInfo->bankName,
            $signupRequest->beneficiaryInfo->branchName,
            $signupRequest->beneficiaryInfo->accountNumber,
        );
    }
}
