{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    gnumake
    jq
    python312
    ruff
    uv
  ];
  shellHook = ''
    export UV_PYTHON_DOWNLOADS=never
    export UV_PYTHON=${pkgs.python312}/bin/python3
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}:''${LD_LIBRARY_PATH:-}
  '';
}
