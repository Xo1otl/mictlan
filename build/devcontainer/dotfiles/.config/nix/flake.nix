{
  description = "Dockerfileでコピーしてnix profile installする用";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      packages.${system}.default = pkgs.buildEnv {
        name = "mictlan-devcontainer";
        paths = [
          pkgs.nix-direnv
        ];
      };
    };
}
