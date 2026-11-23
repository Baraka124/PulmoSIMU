#Requires -Version 5.1
<#
PulmoSIMU A2: verify exact CT/GLB checksums, then (only with -Publish)
upload original CT + 2 GLBs + A2 QA as PUBLIC GitHub Release assets.
The connected ChatGPT GitHub integration cannot upload Release binaries.
Copy the source CT, both GLBs, and local A2 QA JSON files to this directory.
First run without -Publish, inspect results and confirm licensing/de-ID.
#>
[CmdletBinding()]
param([switch]$Publish)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = 'Baraka124/PulmoSIMU'
$tag = 'pulmosimu-a2-nlst-research-2026-11'
$assets = [ordered]@{
 'NLST_Chest_CT_150_Slices_R21(1).nii' = '5b234ac92a9a5d58d15f3f43144d31612d10ff6e60777d8c273cd52b36b13087'
 'PulmoSIMU_NLST_CT_Lung_Quest_Research.glb' = '5e4457003b45f9d7f78e87239f935c4299e4c3b947a9a3ffda0e273216228a0f'
 'PulmoSIMU_A2_Candidate_RAS_Laterality.glb' = '8f8bea9a95fc21a44ca10244c91c249d2836539bf119973bcb18c614c9886d2f'
}
foreach($name in $assets.Keys){
 $file = Join-Path $here $name
 if(-not (Test-Path -LiteralPath $file -PathType Leaf)){throw "Missing binary asset: $name"}
 $digest=(Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
 if($digest -ne $assets[$name]){throw "SHA-256 mismatch: $name"}
 Write-Host "Verified: $name" -ForegroundColor Cyan
}
$manifest=Join-Path $here 'a2-candidate-manifest.json'
$qa=Join-Path $here 'a2-candidate-qa.json'
$vox=Join-Path $here 'a2-candidate-voxel-qa.json'
foreach($file in @($manifest,$qa,$vox)){
 if(-not(Test-Path -LiteralPath $file -PathType Leaf)){throw "Missing A2 QA: $file"}
}
if(-not $Publish){
 Write-Host 'DRY RUN OK — NO PUBLIC UPLOAD.' -ForegroundColor Yellow
 Write-Host 'Verify exact source scan TCIA series provenance, licensing and full de-identification first.'
 exit 0
}
if(-not (Get-Command gh -ErrorAction SilentlyContinue)){
 throw 'Install GitHub CLI (gh) from https://cli.github.com and sign in.'
}
& gh auth status --hostname github.com
if($LASTEXITCODE -ne 0){throw 'Run gh auth login locally. Never share access tokens.'}
& gh release view $tag --repo $repo --json url 2>$null
if($LASTEXITCODE -eq 0){throw "Refusing to overwrite existing release: $tag"}
$confirmed = Read-Host 'Type PROVENANCE-VERIFIED after exact NLST case/licence/de-ID audit'
if($confirmed -cne 'PROVENANCE-VERIFIED'){throw 'No public publication performed.'}
$confirmed = Read-Host 'Type PUBLISH-RESEARCH-ONLY for PUBLIC research assets'
if($confirmed -cne 'PUBLISH-RESEARCH-ONLY'){throw 'No public publication performed.'}
$files = @()
foreach($name in $assets.Keys){$files += (Join-Path $here $name)}
$files += @($manifest,$qa,$vox)
& gh release create $tag @files --repo $repo --target 'feature/mechanistic-lung-foundation' --title 'PulmoSIMU A2 NLST research anatomy — UNVALIDATED' --notes 'Research only: NIfTI source from claimed public TCIA NLST, bilateral airspace meshes, RAS node names corrected in a separate GLB; residual 16.3mL overlapping mask, no verified lobes/airway/vessels, CT-linked procedure BLOCKED. Attribution: TCIA NLST DOI 10.7937/TCIA.HMQ8-J677 (CC BY 4.0, contingent on case provenance).' --prerelease
if($LASTEXITCODE -ne 0){throw 'Upload failed; inspect Release assets before retrying.'}
& gh release view $tag --repo $repo --json url,assets
if($LASTEXITCODE -ne 0){throw 'Unable to verify public Release assets.'}
