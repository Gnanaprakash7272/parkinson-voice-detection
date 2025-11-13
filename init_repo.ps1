$gitPath = "C:\Program Files\Git\bin\git.exe"

# Configure git
& $gitPath config --global user.name "Parkinson Voice Detection"
& $gitPath config --global user.email "user@example.com"

# Initialize repo
cd D:\hacksphere
& $gitPath init
& $gitPath add .
& $gitPath commit -m "Initial commit - Parkinson voice detection app"
& $gitPath branch -M main

Write-Host ""
Write-Host "✅ Git repository initialized and committed!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Go to https://github.com/new and create a new repository"
Write-Host "2. Copy the HTTPS URL (e.g., https://github.com/your-username/my-repo.git)"
Write-Host "3. Run the following command to add the remote and push:"
Write-Host ""
Write-Host "   git remote add origin <your-repo-url>"
Write-Host "   git push -u origin main"
Write-Host ""
