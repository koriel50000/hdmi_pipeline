param(
    [Parameter(Mandatory=$true)]
    [string]$InputGif,

    [string]$OutputBin = "sprite_argb.bin"
)

Add-Type -AssemblyName System.Drawing

$bitmap = $null

try {
    # PowerShellのカレントディレクトリを基準に入力ファイルを解決
    $resolvedPath = Resolve-Path -LiteralPath $InputGif -ErrorAction Stop
    $fullPath = $resolvedPath.ProviderPath

    Write-Host "Input: $fullPath"

    # GIFを読み込む
    $bitmap = [System.Drawing.Image]::FromFile($fullPath)

    # フレーム情報
    $frameDimension = New-Object System.Drawing.Imaging.FrameDimension(
        $bitmap.FrameDimensionsList[0]
    )

    # 1フレーム目を選択
    $bitmap.SelectActiveFrame($frameDimension, 0)

    $width  = $bitmap.Width
    $height = $bitmap.Height

    Write-Host "Size : ${width}x${height}"
    Write-Host "Frame: 0"

    if ($width -ne 500 -or $height -ne 500) {
        Write-Warning "画像サイズが500x500ではありません。"
    }

    # 出力ファイルもカレントディレクトリに作成
    $outputPath = Join-Path (Get-Location) $OutputBin

    $stream = [System.IO.File]::Open(
        $outputPath,
        [System.IO.FileMode]::Create,
        [System.IO.FileAccess]::Write
    )

    try {
        $writer = New-Object System.IO.BinaryWriter($stream)

        try {
            for ($y = 0; $y -lt $height; $y++) {
                for ($x = 0; $x -lt $width; $x++) {

                    $color = $bitmap.GetPixel($x, $y)

                    # 32bit ARGB (0xAARRGGBB)
                    $writer.Write($color.ToArgb())
                }
            }
        }
        finally {
            $writer.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }

    $size = (Get-Item $outputPath).Length

    Write-Host ""
    Write-Host "Completed."
    Write-Host "Output : $outputPath"
    Write-Host "Pixels : $($width * $height)"
    Write-Host "Bytes  : $size"
}
finally {
    if ($null -ne $bitmap) {
        $bitmap.Dispose()
    }
}