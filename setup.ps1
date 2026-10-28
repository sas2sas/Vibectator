# Установка окружения для локальной диктовки (Windows, Python 3.10+).
# Запуск: правый клик -> "Выполнить с помощью PowerShell".
# Модель (~420 МБ) скачивается отдельно, интернет нужен только на этом шаге.

$ErrorActionPreference = 'Stop'
$dir = $PSScriptRoot
Set-Location $dir

Write-Host '== 1/4 Проверка Python =='
$py = $null
foreach ($cmd in @('py -3.12', 'py -3', 'python')) {
    try {
        $path = (& cmd /c "$cmd -c `"import sys;print(sys.executable)`"" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $path) { $py = $path.Trim(); break }
    } catch { }
}
if (-not $py) { throw 'Python не найден. Установите Python 3.10-3.12 и повторите.' }
Write-Host "Python: $py"

Write-Host '== 2/4 Виртуальное окружение =='
$venv = Join-Path $dir '.venv-gigaam'
if (-not (Test-Path $venv)) { & $py -m venv $venv }
$vpy = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $vpy)) { throw "Не удалось создать $venv" }

Write-Host '== 3/4 Зависимости =='
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
# hydra-core тянет старый antlr, который не собирается на Python 3.12
& $vpy -m pip install --no-deps hydra-core==1.3.8 antlr4-python3-runtime==4.9.3 omegaconf==2.3.0
& $vpy -m pip install transformers==4.57.1 sentencepiece soundfile numpy --quiet
& $vpy -m pip install faster-whisper==1.2.0 av==12.3.0 sounddevice keyboard pyperclip pystray pillow requests --quiet

Write-Host '== 4/4 Модель GigaAM-v3 (~420 МБ) =='
if (Test-Path (Join-Path $dir 'models\gigaam')) {
    Write-Host 'Уже скачана, пропускаю.'
} else {
    & $vpy -c @"
import os
from transformers import AutoModel
here = os.path.dirname(os.path.abspath('.'))
model = AutoModel.from_pretrained('ai-sage/GigaAM-v3', revision='e2e_ctc',
                                  trust_remote_code=True, cache_dir=os.path.join(here, 'models', 'gigaam'))
print('модель загружена')
"@
}

Write-Host ''
Write-Host 'Готово. Дальше:'
Write-Host '  1) запустите run.cmd от имени администратора (обязательно)'
Write-Host '  2) держите клавишу Ё, говорите, отпустите'
Write-Host 'Настройки: config.json, диагностика: dictation.log'
