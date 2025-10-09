# evaluator検証

* **着目した点**: 論文に以下のように書かれていた点に着目した
    * `PyTorch code could potentially enhance equation discovery by leveraging differentiable parameter optimization in future.`
* **行った検証**: 
    * jax.numpyを使ってadamで学習率3e-4 epoch 10000でevaluateを試した 
    * jaxopt の bfgs も試した
    * scicpy の l-bfgs-b と bfgs を試した
    * LLMにはimport numpyと伝えて、実際の名前空間にはjax.numpyを設定して関数同定を行った
* **何が明確になったか**: 
    * jaxを使ってbactgrowとoscillator1の関数同定に成功しているので、勾配情報を使ったその他の最適化手法も使えることが明確になった
    * パラメータが10個の時はlbfgsの方が速度も精度も良いことが明確になった
    * `pilot_study.ipynb` に書いてる非線形結合モード方程式の近似式の係数がscipyのbfgsでしか収束しない、ラインサーチの洗練度合いのせいかもしれない
    * 係数同定の精度はとても重要であることが明確になった
    * どのオプティマイザでも係数が決められない関数がけっこうあることがわかった、5個しか試してないのに決定できないの2つ
