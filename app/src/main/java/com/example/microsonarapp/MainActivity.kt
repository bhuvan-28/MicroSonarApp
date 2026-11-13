package com.example.microsonarapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var sonarEngine: SonarEngine
    private lateinit var tvDimensions: TextView
    private lateinit var tvVolume: TextView
    private lateinit var tvMaterial: TextView
    private lateinit var btnPing: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvDimensions = findViewById(R.id.tvDimensions)
        tvVolume = findViewById(R.id.tvVolume)
        tvMaterial = findViewById(R.id.tvMaterial)
        btnPing = findViewById(R.id.btnPing)

        // Request Audio Recording Permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 101)
        }

        sonarEngine = SonarEngine(this)

        btnPing.setOnClickListener {
            btnPing.isEnabled = false
            btnPing.text = "LISTENING..."

            lifecycleScope.launch {
                try {
                    val metrics = sonarEngine.runSweep()
                    tvDimensions.text = String.format("%.2fm × %.2fm × %.2fm", metrics.width, metrics.length, metrics.height)
                    tvVolume.text = String.format("Room Volume: %.1f m³", metrics.volume)
                    tvMaterial.text = "Material: ${metrics.material}"
                } catch (e: Exception) {
                    Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                } finally {
                    btnPing.isEnabled = true
                    btnPing.text = "PULSE SONAR"
                }
            }
        }
    }
}