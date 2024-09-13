<?php

namespace voice;

class Controller
{
    private App $app;

    public function __construct(App $app)
    {
        $this->app = $app;
    }

    public function editTag(array $postData, \common\Id $accountId)
    {
        $input = new EditTagInput($accountId, new \common\Id($postData['voice_id']), $postData['voice_title'], $postData['age_tag'], $postData['character_tag']);
        $this->app->editTag($input);
    }

    public function upload(Storage $storage, array $postData, array $fileData, \common\Id $accountId)
    {
        $voiceFile = $fileData['audio'];
        if ($voiceFile['error'] !== UPLOAD_ERR_OK) {
            \logger\fatal($voiceFile['error']);
        }

        $filename = pathinfo($voiceFile['tmp_name'], PATHINFO_FILENAME);
        $mimeType = $voiceFile['type'];
        $size = $voiceFile['size'];
        $createdAt = new \DateTime();
        $title = $postData['voiceTitle'];
        $input = new Input($accountId, $title, $filename, $mimeType, $size, $createdAt);

        $storage->uploadVoice($voiceFile['tmp_name'], $input);

        $this->app->upload($input);
    }

    /**
     * @param \common\Id $accountId
     * @return Voice[]
     */
    public function getAll(\common\Id $accountId): array
    {
        return $this->app->getAll($accountId);
    }

    public function deleteVoice(array $postData, \common\Id $accountId)
    {
        $this->app->delete(new \common\Id($postData['voice_id']), $accountId);
    }
}
