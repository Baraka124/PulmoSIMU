#Requires -Version 5.1
<#
.SYNOPSIS
  Verify and publish the PulmoSIMU NLST A1 research CT/GLB as public GitHub Release assets.
.DESCRIPTION
  Does not publish or modify the source until explicitly run on the user's machine.
  Verifies original SHA-256 digests, opt-in permissions and Github CLI access.
  Publishes a prerelease attached to feature/mechanistic-lung-foundation rather
  than adding a 75 MiB binary to normal Git history.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$SourceNifti,
  [Parameter(Mandatory=$true)][string]$MeshGlb,
  [switch]$Publish
)
$ErrorActionPreference = 'Stop'
$repo = 'Baraka124/PulmoSIMU'
$tag = 'a1-nlst-ct-research-v1'
$target = 'feature/mechanistic-lung-foundation'
$expectedNifti = '5b234ac92a9a5d58d15f3f43144d31612d10ff6e60777d8c273cd52b36b13087'
$expectedMesh = '5e4457003b45f9d7f78e87239f935c4299e4c3b947a9a3ffda0e273216228a0f'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Join-Path $here 'study-manifest.json'
$a2report = Join-Path $here 'a2-laterality-qa.json'
if(-not (Test-Path -LiteralPath $a2report -PathType Leaf)) { throw 'Missing mandatory A2 laterality QA report.' }
if(-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
  throw "Missing local study-manifest.json. Download the folder from the feature branch first."
}
$source = (Resolve-Path -LiteralPath $SourceNifti).Path
$mesh = (Resolve-Path -LiteralPath $MeshGlb).Path
function Verify-Hash([string]$File,[string]$Expected) {
  $actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
  if($actual -ne $Expected) { throw "SHA-256 mismatch for $File. Refusing public publication." }
  Write-Host "Verified SHA-256: $(Split-Path -Leaf $File)" -ForegroundColor Cyan
}
Verify-Hash $source $expectedNifti
Verify-Hash $mesh $expectedMesh
if(-not $Publish) {
  Write-Host 'Dry run complete. No files were uploaded.' -ForegroundColor Yellow
  Write-Host 'For public distribution, rerun with -Publish after checking original NLST CC BY 4.0 provenance and de-identification.'
  exit 0
}
if(-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw 'GitHub CLI (gh) is not installed. Get it from https://cli.github.com/'
}
& gh auth status -h github.com
if($LASTEXITCODE -ne 0){throw 'GitHub CLI authentication is required.'}
Write-Host 'This publishes the CT, its derived GLB, and manifest as publicly accessible Release assets.'
$confirmation = Read-Host "Type PUBLISH-NLST-A1 to confirm you verified license and de-identification"
if($confirmation -cne 'PUBLISH-NLST-A1'){throw 'No public upload performed.'}
$notes = @'
PRELIMINARY RESEARCH DATA — NOT A CLINICALLY VALIDATED LUNG MODEL
CRITICAL A2 QA FAILURE: source mesh LEFT/RIGHT names are reversed relative to
NIfTI RAS anatomy. See a2-laterality-qa.json. Do not connect this GLB to
patient-specific bronchoscopic navigation until independently corrected.

Source: National Lung Screening Trial (NLST), The Cancer Imaging Archive.
DOI: https://doi.org/10.7937/TCIA.HMQ8-J677
License: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/

Derived NIfTI and lung-surface GLB are converted / segmented research data.
Original source series has not been independently reidentified.
Derived segmentation may have right-sided leakage, missing airspace, no verified
lobes/airways/vessels, and no source-linked bronchoscopy. See study-manifest.json.
Use only as a research reference; no patient-specific clinical decisions.
'@
$notesPath = Join-Path ([IO.Path]::GetTempPath()) ('pulmosimu-nlst-release-'+[guid]::NewGuid().ToString('N')+'.md')
try {
  [IO.File]::WriteAllText($notesPath,$notes)
  & gh release view $tag --repo $repo --json url 2>$null
  if($LASTEXITCODE -eq 0){
    throw "Release $tag already exists. Refusing to overwrite verified study assets; review the existing release first."
  }
  & gh release create $tag $source $mesh $manifest $a2report --repo $repo --target $target --title 'PulmoSIMU A1 — NLST CT research geometry (unvalidated)' --notes-file $notesPath --prerelease --latest=false
  if($LASTEXITCODE -ne 0){throw 'GitHub release upload failed; inspect draft release/assets before retrying.'}
  & gh release view $tag --repo $repo --json url,assets
  if($LASTEXITCODE -ne 0){throw 'Unable to verify public release assets.'}
} finally {
  Remove-Item -LiteralPath $notesPath -Force -ErrorAction SilentlyContinue
}
