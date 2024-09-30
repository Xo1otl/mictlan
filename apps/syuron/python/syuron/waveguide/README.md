## 波長とデバイス構造を決める(パラメータを振る)

- 42行目で波長の配列
- 45行目でコアの幅の配列
- 48行目で2層目の厚さの配列
- 51行目で3層目の厚さの配列
- 54行目で温度は20どこてい
- 固定された計算領域を決める
    - 93~100で基板厚さクラッドの厚さリッジの高さ
- 全通り計算します

## 各層,各材料の屈折率を入力する

- 屈折率が波長に依存する
- 誘電率テンソルと材料由来のいくつかの定数が決まっている
- それぞれの層について決める

## 誘電率の分布を導出する

- 材料などの屈折率と、デバイス構造から導出
- 118行目から136行目まで
- 断面だけを決めれる
- 簡単なものはwaveguidemeshfull関数で導出する
    - デバイスの5層構造を定義したら求まる
    - 5層構造
    - 1層目は基板、パラメータは
    - 2,3,4層が導波路のコア(伝搬するところ)
    - 5がクラッド(基板と同じ)
- いろいろ調整できるのでシンプルな構造から始めよう
- シンプルな構造考えたい

## 磁界(電解)分布を求める

- WGMODESを使う
- 境界条件と波長と誘電率分布からマクスウェル方程式で求める、複数の磁界分布の可能性がある
- 固有値が実行屈折率で、固有ベクトルが磁界分布(すぐ`電界分布`に変換できる)
- 何個求めるかを決めることができる、全部求めるのは現実的ではない
- 境界条件はデフォルトのままで大丈夫
- 133行目でやってる

## 磁界分布の分類

- 151行目まで
- 一つの誘電率分布に存在できる電解分布は大量にあり、求める個数を自分で決めている
- それらのモードを分類する
- 分類不能な電解分布を除く
- モードは偏向と次数で決まる
- TMかTEか分類する: 偏向(`polarization`)
- `modeindex`で次数が決まる

## 求まった磁界分布から完全な電解分布を導出する

- 分類した内で、分類可能なものに対して導出を行う
- モードソルバーのpostprocessという関数で求まる
    - デバイス構造と磁界分布で求まる
    
## 電界分布の規格化をする

- 158行目
- 電磁界が運ぶパワーが1になるように規格化
- `normalize`関数使うだけ

## 分類後の電界から任意の電界を抽出する

- どういうのが欲しいのか決めるパラメータ
    - fs = 1が基本波, fs = 2がsh波(λ/2)
    - pol = 1の時TE, pol = 2の時TM
    - mode_xは電界分布のx軸方向の腹の数, mode_yはy軸方向の腹の数
- 入れる光のモードだけ考えればいい

## ソースコード

```matlab
% 注意！計算を実行すると現在のワークスペースが現在時刻の名前で保存されたのち削除されます。
addpath('data\');
save('data\archive\'+string(datetime('now'),'yyyy年MM月dd日HH時mm分ss秒'));
clear;

% ファイル名を入力するようにポップアップが出るので忘れずに入力すること
prompt = {'ファイル名'};
dlgtitle = 'Input';
dims = [1 35];
definput = {''};
answer = inputdlg(prompt,dlgtitle,dims,definput);
input_filename = string(answer(1,1));

start_time = datetime('now');
% フォルダ名を指定するとワークスペースが自動で'workspace'の名前で保存される
foldername = string(datetime('now'),'yyyy年MM月dd日HH時mm分ss秒')+'_'+input_filename;
mfolder = append('data\', foldername);
mkdir(mfolder);

% フォルダのパスを追加
addpath('lib\');                                 %lib をパスに加える
addpath('lib\refractiveindex\');                 %lib をパスに加える

% 物理常数
c = 2.99792458*10^14;                            %光速
e = 8.854*10^(-18);                              %真空の誘電率

% 以下各自の計算に合わせて変更すること
d33_core = 6.3*10^(-6);                          %非線形光学定数d33

% Tips:高次モードを計算するには計算するモード総数を増やす必要あり
nmodes  =  [5, 50];                            % 計算するモード総数
nmode_x = 5;                                     % TEMxy の xモード総数 5
nmode_y = 5;                                     % TEMxy の yモード総数 5

dx = 0.01;                                       % グリッドサイズ（水平）
dy = 0.001;                                       % グリッドサイズ（垂直）

% 変化させたいパラメータを配列として設定
% 等差数列なら'1:0.5:4;'のように。この場合は1から4まで0.5ずつ増加する等差数列が生成される。場合によっては対数配列を用意したり任意の配列を手入力すると余計な計算を減らすことができる。
% Tips:変化させる必要のないパラメータのstartとendは1にしてfor中の定数を直接編集するのがおすすめ
wls = 420:10:500;  %計算したい最短波長
nwl = length(wls);

ws = 0.60;  %計算したい最小導波路 注意!単位は1um 細かいことは'rw'で検索して確かめてください
nw = length(ws);

h2s = 0.358;  %計算したい下層コアの最小膜厚
nh2 = length(h2s);  %計算したい下層コア膜厚数

h3s = 0;  %計算したい下層コアの最小膜厚
nh3 = length(h3s);  %計算したい下層コア膜厚数

ts = 20;
nt = length(ts);  %計算したい温度

%事前に0で埋められた配列を用意することで計算を軽くする
%ここにかかれたものがすべてではないので必要に応じて書き足す
neff_f11        = zeros(nwl, nw, nh2, nh3, nt);                      % 実効屈折率     f11
neff_f21        = zeros(nwl, nw, nh2, nh3, nt);                      % 実効屈折率     f11
neff_f31        = zeros(nwl, nw, nh2, nh3, nt);                      % 実効屈折率     f11
neff_s11        = zeros(nwl, nw, nh2, nh3, nt);                      % 実効屈折率     s11
neff_s12        = zeros(nwl, nw, nh2, nh3, nt);                      % 実効屈折率     s12
neff_s13        = zeros(nwl, nw, nh2, nh3, nt);                      % 実効屈折率     s13
neff_s14        = zeros(nwl, nw, nh2, nh3, nt);                      % 実効屈折率     s14
neff_s15        = zeros(nwl, nw, nh2, nh3, nt);                      % 実効屈折率     s15

V11_1layer      = zeros(nwl, nw, nh2, nh3, nt);                      % 重なり積分     f11 s12
V12_1layer      = zeros(nwl, nw, nh2, nh3, nt);                      % 重なり積分     f11 s12
V13_1layer      = zeros(nwl, nw, nh2, nh3, nt);                       % 重なり積分     f11 s13
V14_1layer      = zeros(nwl, nw, nh2, nh3, nt);                      % 重なり積分     f11 s13（分極反転あり・二層・上側を反転）
V15_1layer      = zeros(nwl, nw, nh2, nh3, nt);                      % 重なり積分     f11 s13（分極反転あり・二層・下側を反転）

kappa_s11       = zeros(nwl, nw, nh2, nh3, nt);
kappa_s12       = zeros(nwl, nw, nh2, nh3, nt);
kappa_s13       = zeros(nwl, nw, nh2, nh3, nt);
kappa_s14       = zeros(nwl, nw, nh2, nh3, nt);
kappa_s15       = zeros(nwl, nw, nh2, nh3, nt);


% Tips:繰り返し数の多い変数をparforに書き換えて並列計算をすると早くなる。タスクマネージャでCPUの各スレットの使用量が増えるはず。通常のforでは１つのスレットでしか計算しない。
% 注意!parforは複数使えないので最も並列化して効果が得られそうなもののみ書き換えること。
parfor iwl= 1:nwl
    for iw= 1:nw
        for ih2 = 1:nh2
           for ih3= 1:nh3
               for it = 1:nt


                    lamda_base = 0.001*wls(iwl);                               %基本波の波長

                    % チャネル導波路構造の定義〔ストリップ装荷チャネル導波路〕
                    rw = ws(iw)/2;       	% ストリップ幅（半幅）
                    h1 = 1;                   % 下部クラッド厚
                    h2 = h2s(ih2);         % スラブ厚
                    h3 = h3s(ih3);                  % ストリップ厚
                    h4 = 0;       % 上部クラッド厚
                    h5 = 1;                   % 上部クラッド厚
                    sh = h2;              % ストリップ厚
                    side = 0.5;                 % ストリップ脇の計算幅

                    T = ts(it);

                    mode_wpxy = zeros(2, 2, nmode_x, nmode_y);	% 変換テーブル（高調波/基本波, 偏光, xモード次数, yモード次数） → 固有値番号
                    corr_wpxy = zeros(2, 2, nmode_x, nmode_y);	% 変換テーブル（高調波/基本波, 偏光, xモード次数, yモード次数） → モード相関
                    neff_wpxy = NaN(2, 2, nmode_x, nmode_y);	% 変換テーブル（高調波/基本波, 偏光, xモード次数, yモード次数） → 実効屈折率

                    % 導波路の構造メッシュをもとに一時格納用の配列を宣言する
                    % 'waveguidemeshfull'は配布プログラムのオリジナルなので詳細は関数の冒頭に英語で記述あり。関数名を右クリックで開いてみてください。
                    [x,y,xc,yc,nx,ny,epszz,edges] = waveguidemeshfull([1,1,1,1,1],[h1,h2,h3,h4,h5],sh,rw,side,dx,dy); % 空打ち（nx，ny算出のため）
                    ex_wpxy = NaN(nx, ny, 2, 2, nmode_x, nmode_y);      % 電場（位置, 高調波/基本波, 偏光, xモード次数, yモード次数）
                    ey_wpxy = NaN(nx, ny, 2, 2, nmode_x, nmode_y);      % 電場（位置, 高調波/基本波, 偏光, xモード次数, yモード次数）
                    ez_wpxy = NaN(nx, ny, 2, 2, nmode_x, nmode_y);      % 電場（位置, 高調波/基本波, 偏光, xモード次数, yモード次数）
                    hx_wpxy = NaN(nx+1, ny+1, 2, 2, nmode_x, nmode_y);  % 磁場（位置, 高調波/基本波, 偏光, xモード次数, yモード次数）
                    hy_wpxy = NaN(nx+1, ny+1, 2, 2, nmode_x, nmode_y);  % 磁場（位置, 高調波/基本波, 偏光, xモード次数, yモード次数）
                    hz_wpxy = NaN(nx+1, ny+1, 2, 2, nmode_x, nmode_y);  % 磁場（位置, 高調波/基本波, 偏光, xモード次数, yモード次数）

                    for fs= 1:2                                         % 波長をふる（ωと2ω）
                        % 誘電率分布の定義
                        % 各軸ごとに毎回誘電率のメッシュを'waveguidemeshfull'で定義
                        lambda = lamda_base/fs; dn = 0; theta = 0;
                        [e1xx, e1yy, e1xy, e1yx, e1zz, e2xx, e2yy, e2xy, e2yx, e2zz, e3xx, e3yy, e3xy, e3yx, e3zz, e4xx,e4yy,e4xy,e4yx,e4zz,e5xx,e5yy,e5xy,e5yx,e5zz, guess] = define_eps_SL(lambda, dn, theta,T);  % 誘電率テンソルの定義
                        [~,~,~,~,~,~,epsxx,~] = waveguidemeshfull([sqrt(e1xx),sqrt(e2xx),sqrt(e3xx),sqrt(e4xx),sqrt(e5xx)],[h1,h2,h3,h4,h5],sh,rw,side,dx,dy);  % メッシュ生成
                        [~,~,~,~,~,~,epsxy,~] = waveguidemeshfull([sqrt(e1xy),sqrt(e2xy),sqrt(e3xy),sqrt(e4xy),sqrt(e5xy)],[h1,h2,h3,h4,h5],sh,rw,side,dx,dy);
                        [~,~,~,~,~,~,epsyx,~] = waveguidemeshfull([sqrt(e1yx),sqrt(e2yx),sqrt(e3yx),sqrt(e4yx),sqrt(e5yx)],[h1,h2,h3,h4,h5],sh,rw,side,dx,dy);
                        [~,~,~,~,~,~,epsyy,~] = waveguidemeshfull([sqrt(e1yy),sqrt(e2yy),sqrt(e3yy),sqrt(e4yy),sqrt(e5yy)],[h1,h2,h3,h4,h5],sh,rw,side,dx,dy);
                        [x,y,~,~,nx,ny,epszz,edges] = waveguidemeshfull([sqrt(e1zz),sqrt(e2zz),sqrt(e3zz),sqrt(e4zz),sqrt(e5zz)],[h1,h2,h3,h4,h5],sh,rw,side,dx,dy);
                        [x,y,xc,yc,ddx,ddy] = stretchmesh(x,y,[10,10,40,40],[1,1,2,2]);    % 境界部のメッシュをストレッチ

                        % モードを求解
                        % 'wgmodes'がモードソルバーの中心的関数。これも配布プログラムのオリジナルなので詳細は関数の冒頭に記述あり。必要があれば論文を読むこと。
                        guess=e2yy;
                        [Hx,Hy,neff] = wgmodes (lambda, guess, nmodes(fs), ddx, ddy, epsxx, epsyy, epszz, '0000');
                        ex = NaN(size(Hx,1),size(Hx,2)); ey = NaN(size(Hx,1),size(Hx,2)); ez = NaN(size(Hx,1),size(Hx,2));
                        hx = NaN(size(Hx,1),size(Hx,2)); hy = NaN(size(Hx,1),size(Hx,2)); hz = NaN(size(Hx,1),size(Hx,2));
                        H0 = zeros(size(Hx,1),size(Hx,2));                      % ダミーの磁界分布（ゼロ）

                        % モードの分類
                        for ii = nmodes(fs):-1:1                                       % 固有値は実効屈折率の高い順に求まる （固有値番号 ii）
                            [pol] = polarization(dx,dy,Hx(:,:,ii),Hy(:,:,ii));          % 偏光（quasi-TE か quasi-TM）を判定
                            if pol == 1
                                [mode_x, mode_y] = modeindex(Hy(:,:,ii), epszz, e2zz);  % quasi-TE としてモード次数 mode_x, mode_y を判定
                                e1 = e1xx;
                                e5 = e5xx;
                            else
                                [mode_x, mode_y] = modeindex(Hx(:,:,ii), epszz, e2zz);  % quasi-TM としてモード次数 mode_x, mode_y を判定
                                e1 = e1yy;
                                e5 = e5yy;
                            end

                            if (neff(ii)>=sqrt(max(e1,e5))) && (mode_x <= nmode_x) && (mode_y <= nmode_y)    % 導波モード以外の実効屈折率を NaN とする
                                mode_wpxy(fs, pol, mode_x, mode_y) = ii;          	% 偏光 pol と モード次数 mode_x, mode_y から 固有値番号 ii を返す参照テーブル
                                %                    corr_wpxy(fs, pol, mode_x, mode_y) = correlation;
                                neff_wpxy(fs, pol, mode_x, mode_y) = real(neff(ii)); 	% 偏光 pol と モード次数 mode_x, mode_y から 実効屈折率 を返す参照テーブル
                                neff_temp = real(neff(ii));
                                hx = Hx(:, :, ii); hy = Hy(:, :, ii);

                                [hz,ex,ey,ez] = postprocess (lambda, neff_temp, hx, hy, dx, dy, epsxx, epsyy, epszz, '0000');
                                [ex,ey,ez,hx,hy,hz] = normalize(dx,dy,ex,ey,ez,hx,hy,hz);
                                ex_wpxy(:, :, fs, pol, mode_x, mode_y) = ex(:, :);
                                ey_wpxy(:, :, fs, pol, mode_x, mode_y) = ey(:, :);
                                ez_wpxy(:, :, fs, pol, mode_x, mode_y) = ez(:, :);
                            end
                        end
                    end

                    % 特定のモードについて電界分布と実効屈折率を取り出す
                    fs2 = 1; pol = 2; mode_x = 1; mode_y = 1;
                    neff_f11(iwl, iw, ih2, ih3, it) = squeeze(neff_wpxy(fs2, pol, mode_x, mode_y));
                    ex_f11 = zeros(size(ex_wpxy,1), size(ex_wpxy,2)); ex_f11(:, :) = ex_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ey_f11 = zeros(size(ey_wpxy,1), size(ey_wpxy,2)); ey_f11(:, :) = ey_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ez_f11 = zeros(size(ez_wpxy,1), size(ez_wpxy,2)); ez_f11(:, :) = ez_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    savemode(ey_f11, mfolder,'TM00',lamda_base,rw*2,h2,h3,T);

                    fs2 = 1; pol = 2; mode_x = 2; mode_y = 1;
                    neff_f21(iwl, iw, ih2, ih3, it) = squeeze(neff_wpxy(fs2, pol, mode_x, mode_y));
                    ex_f21 = zeros(size(ex_wpxy,1), size(ex_wpxy,2)); ex_f21(:, :) = ex_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ey_f21 = zeros(size(ey_wpxy,1), size(ey_wpxy,2)); ey_f21(:, :) = ey_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ez_f21 = zeros(size(ez_wpxy,1), size(ez_wpxy,2)); ez_f21(:, :) = ez_wpxy(:, :, fs2, pol, mode_x, mode_y);

                    fs2 = 1; pol = 2; mode_x = 3; mode_y = 1;
                    neff_f31(iwl, iw, ih2, ih3, it) = squeeze(neff_wpxy(fs2, pol, mode_x, mode_y));
                    ex_f31 = zeros(size(ex_wpxy,1), size(ex_wpxy,2)); ex_f31(:, :) = ex_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ey_f31 = zeros(size(ey_wpxy,1), size(ey_wpxy,2)); ey_f31(:, :) = ey_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ez_f31 = zeros(size(ez_wpxy,1), size(ez_wpxy,2)); ez_f31(:, :) = ez_wpxy(:, :, fs2, pol, mode_x, mode_y);

                    %[~,~,~,~,~,~,ddd,~] = waveguidemeshfull_for_d33([0,d33_core,-1*d33_core,0,0],[h1,0.210,0.144,h4,h5],sh,rw,side,dx,dy); %非線形光学常数のマップを生成
                    fs2 = 2; pol = 2; mode_x = 1; mode_y = 1;
                    neff_s11(iwl, iw, ih2, ih3, it) = squeeze(neff_wpxy(fs2, pol, mode_x, mode_y));
                    ex_s11 = zeros(size(ex_wpxy,1), size(ex_wpxy,2)); ex_s11(:, :) = ex_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ey_s11 = zeros(size(ey_wpxy,1), size(ey_wpxy,2)); ey_s11(:, :) = ey_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ez_s11 = zeros(size(ez_wpxy,1), size(ez_wpxy,2)); ez_s11(:, :) = ez_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    %[~, ol_lin] = overlap(dx, dy, ey_f11,ey_s11, ddd); kappa_s11(iwl, iw, ih2, ih3, it) = abs(2*2*pi*c/lamda_base*e*ol_lin/4)*10.^6;

                    fs2 = 2; pol = 2; mode_x = 1; mode_y = 2;
                    neff_s12(iwl, iw, ih2, ih3, it) = squeeze(neff_wpxy(fs2, pol, mode_x, mode_y));
                    ex_s12 = zeros(size(ex_wpxy,1), size(ex_wpxy,2)); ex_s12(:, :) = ex_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ey_s12 = zeros(size(ey_wpxy,1), size(ey_wpxy,2)); ey_s12(:, :) = ey_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    %[~, ol_lin] = overlap(dx, dy, ey_f11,ey_s12, ddd); kappa_s12(iwl, iw, ih2, ih3, it) = abs(2*2*pi*c/lamda_base*e*ol_lin/4)*10.^6;

                    fs2 = 2; pol = 2; mode_x = 1; mode_y = 3;
                    neff_s13(iwl, iw, ih2, ih3, it) = squeeze(neff_wpxy(fs2, pol, mode_x, mode_y));
                    ex_s13 = zeros(size(ex_wpxy,1), size(ex_wpxy,2)); ex_s13(:, :) = ex_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ey_s13 = zeros(size(ey_wpxy,1), size(ey_wpxy,2)); ey_s13(:, :) = ey_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    %[~, ol_lin] = overlap(dx, dy, ey_f11,ey_s13, ddd); kappa_s13(iwl, iw, ih2, ih3, it) = abs(2*2*pi*c/lamda_base*e*ol_lin/4)*10.^6;

                    savemode(ey_s13, mfolder,'TM02',lamda_base,rw*2,h2,h3,T);

                    fs2 = 2; pol = 2; mode_x = 1; mode_y = 4;
                    neff_s14(iwl, iw, ih2, ih3, it) = squeeze(neff_wpxy(fs2, pol, mode_x, mode_y));
                    ex_s14 = zeros(size(ex_wpxy,1), size(ex_wpxy,2)); ex_s14(:, :) = ex_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ey_s14 = zeros(size(ey_wpxy,1), size(ey_wpxy,2)); ey_s14(:, :) = ey_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    %[~, ol_lin] = overlap(dx, dy, ey_f11,ey_s14, ddd); kappa_s14(iwl, iw, ih2, ih3, it)  = abs(2*2*pi*c/lamda_base*e*ol_lin/4)*10.^6;
                    savemode(ey_s14, mfolder,'TM03',lamda_base,rw*2,h2,h3,T);

                    fs2 = 2; pol = 2; mode_x = 1; mode_y = 5;
                    neff_s15(iwl, iw, ih2, ih3, it) = squeeze(neff_wpxy(fs2, pol, mode_x, mode_y));
                    ex_s15 = zeros(size(ex_wpxy,1), size(ex_wpxy,2)); ex_s15(:, :) = ex_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    ey_s15 = zeros(size(ey_wpxy,1), size(ey_wpxy,2)); ey_s15(:, :) = ey_wpxy(:, :, fs2, pol, mode_x, mode_y);
                    %[ol_log, ol_lin] = overlap(dx, dy, ey_f11,ey_s15, ddd); kappa_s15(iwl, iw, ih2, ih3, it)  = abs(2*2*pi*c/lamda_base*e*ol_lin/4)*10.^6;
                    savemode(ey_s15, mfolder,'TM04',lamda_base,rw*2,h2,h3,T);

                    EYSH=ey_s15;    %モードを指定
                    [row_n , col_n]=size(EYSH);
                    reverseposi = 1210;

                    EY_reverse=horzcat(EYSH(:,1:reverseposi)*-1,EYSH(:,reverseposi+1:end));

                    x = dy:dy:ny*dy;

                    if (length(dx) ~= nx),
                      dx2 = dx*ones(nx,1);
                    end
                    if (length(dy) ~= ny),
                      dy2 = dy*ones(1,ny);
                    end

                    SZ = d33_core*real((epsyy==e2yy).*((ey_f11.^2).*EY_reverse));
                    dA = dx2*dy2;
                    ol_log = log((sum(SZ(:).*dA(:)))^2);
                    ol_lin = (sum(SZ(:).*dA(:)));
                    kappa= abs(2*2*pi*c/lamda_base*e*ol_lin/4)*10.^6;
                    once_inver(iwl,1) = kappa/100;
                   
               end

            end
        end
    end
end
%neff_all = [neff_f11.' neff_f21.' neff_f31.' neff_s11.' neff_s12.' neff_s13.' neff_s14.' neff_s15.'];
finish_time = datetime('now');
calc_time = finish_time-start_time;
%{
params = {  '計算日時', string(datetime('now'),'yyyy年MM月dd日HH時mm分ss秒');
            '計算時間', calc_time;
            '波長', wls;
            '導波路幅', ws;
            '下部クラッド', h1;
            '二層目厚さ', h2s;
            '三層目厚さ', h3s;
            '上部クラッド', h5;
            '側部クラッド', side;
            'dx', dx;
            'dy', dy;
            '常光屈折率', sqrt(e1xx),sqrt(e2xx),sqrt(e3xx),sqrt(e5xx);
            '異常光屈折率', sqrt(e1yy),sqrt(e2yy),sqrt(e3yy),sqrt(e5yy);
            }
%}
save(append(mfolder,'\',foldername));
script2html('main_v3_20230121.m',mfolder);
copyfile('lib', append(mfolder,'\lib'));
```
