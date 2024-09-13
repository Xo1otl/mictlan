<?php

namespace mail;

class NotificationService implements \submission\NotificationService
{
    private const MAIL_ENCODING = '7bit';
    private const MAIL_HOST = 's16.valueserver.jp';
    private const MAIL_FROM = 'demo@koemade.net';
    private const MAIL_PHP_LANGUAGE = 'japanese';
    private const MAIL_PHP_INTERNAL_ENCODING = 'UTF-8';
    private const MAIL_CHARSET = 'iso-2022-jp';
    private const MAIL_FROM_NAME = 'コエメイド運営';

    public function notify(\submission\SignupRequest $signupRequest): void
    {
        \logger\info("notify");
        $mail = new \PHPMailer\PHPMailer\PHPMailer(true);
        $mail->SMTPDebug = 4;

        try {
            mb_language(self::MAIL_PHP_LANGUAGE);
            mb_internal_encoding(self::MAIL_PHP_INTERNAL_ENCODING);
            $mail->CharSet = self::MAIL_CHARSET;
            $mail->Encoding = self::MAIL_ENCODING;
            $mail->isSMTP();
            $mail->Host = self::MAIL_HOST;
            $mail->SMTPAuth = false;
            $mail->SMTPSecure = \PHPMailer\PHPMailer\PHPMailer::ENCRYPTION_STARTTLS;
            $mail->Port = 25;

            $mail->setFrom(self::MAIL_FROM, mb_encode_mimeheader(self::MAIL_FROM_NAME));
            $mail->addAddress('demo@koemade.net');
            $mail->addAttachment(__DIR__ . '/../uploads/submission/' . $signupRequest->idImage->getFullname());
            $mail->isHTML(false);
            $mail->Subject = mb_encode_mimeheader('ユーザー登録申請の通知');
            $mail->Body = $this->constructBody($signupRequest);

            $mail->send();
        } catch (\PHPMailer\PHPMailer\Exception $e) {
            \logger\fatal("Message could not be sent. Mailer Error: {$mail->ErrorInfo}");
        }
        $file = __DIR__ . '/../uploads/submission/' . $signupRequest->idImage->getFullname();
        if (file_exists($file)) {
            if (unlink($file)) {
                echo "File deleted successfully.";
            } else {
                echo "Error deleting the file.";
            }
        } else {
            echo "File does not exist.";
        }
    }

    function constructBody(\submission\SignupRequest $signupRequest): string
    {
        $body = mb_convert_encoding('ユーザーからアカウント登録の申請がありました。', 'JIS', self::MAIL_PHP_INTERNAL_ENCODING);
        $body .= "\n";

        $body .= mb_convert_encoding("ID: {$signupRequest->id}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("メールアドレス: {$signupRequest->email}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("名前: {$signupRequest->japaneseName->name}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("ふりがな: {$signupRequest->japaneseName->furigana}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("住所: {$signupRequest->address->text}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("電話番号: {$signupRequest->tel->number}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("自己PR: {$signupRequest->selfPromotion}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("銀行名: {$signupRequest->beneficiaryInfo->bankName}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("支店名: {$signupRequest->beneficiaryInfo->branchName}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("口座番号: {$signupRequest->beneficiaryInfo->accountNumber}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";

        $body .= mb_convert_encoding("ID画像:", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $idImage = $signupRequest->idImage;
        $body .= mb_convert_encoding("ファイル名: {$idImage->filename}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("MIMEタイプ: {$idImage->mimeType}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("サイズ: {$idImage->size} バイト", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";
        $body .= mb_convert_encoding("作成日時: {$idImage->createdAt->format('Y-m-d H:i:s')}", 'JIS', self::MAIL_PHP_INTERNAL_ENCODING) . "\n";

        return $body;
    }
}
