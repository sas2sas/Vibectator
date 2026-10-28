; Установщик Vibectator (Inno Setup 6).
; Сборка: iscc installer.iss  (или двойной клик по build-installer.cmd)

#define MyAppName "Vibectator"
#define MyAppVersion "1.0.0"
#define MyAppExeName "Vibectator.exe"
; Каталог сборки: dist-new используется, пока dist занят запущенной копией
#define BuildDir "dist-new"

[Setup]
AppId={{9C2E4B51-7A3D-4F60-9E1B-2D7C8A5F4B10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Vibectator
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir=installer
OutputBaseFilename=Vibectator-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#BuildDir}\Vibectator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Остановить диктовку"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--stop"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  { Пользовательские настройки и логи остаются в %LOCALAPPDATA%\Vibectator }
end;
