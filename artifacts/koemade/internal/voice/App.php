<?php

namespace voice;

class App
{
    private Repo $repo;

    function __construct(Repo $repo)
    {
        $this->repo = $repo;
    }

    public function upload(Input $input)
    {
        $this->repo->upload($input);
    }

    public function delete(\common\Id $voiceId, \common\Id $accountId)
    {
        $this->repo->deleteById($voiceId, $accountId);
    }

    public function editTag(EditTagInput $input)
    {
        $this->repo->editTag($input);
    }

    /**
     * @param \common\Id $accountId
     * @return Voice[]
     */
    public function getAll(\common\Id $accountId): array
    {
        return $this->repo->getAll($accountId);
    }
}
