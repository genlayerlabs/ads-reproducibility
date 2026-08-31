{
  description = "Pinned numerical environment for Aggregate Disambiguation Systems Part 1";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/48652e9d5aea46e555b3df87354280d4f29cd3a3";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = function:
        nixpkgs.lib.genAttrs systems (system: function nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python313.withPackages (pythonPackages: with pythonPackages; [
              matplotlib
              numpy
              scipy
            ]))
          ];
        };
      });
    };
}
