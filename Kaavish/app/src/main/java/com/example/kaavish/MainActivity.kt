package com.example.kaavish

import android.content.Context
import android.content.pm.ActivityInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.widget.TextView


class MainActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var mSensorManager : SensorManager
    private var mAccelerometer : Sensor ?= null
    private var mGyroscope : Sensor ?= null
    private var resume = false

    override fun onCreate(savedInstanceState: Bundle?) {
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN)

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mSensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        print("accuracy changed")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event != null && resume) {
            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                val accX = event.values[0]
                val temp01:Double = String.format("%.3f", accX).toDouble()
                val temp02:Double = String.format("%.2f", temp01).toDouble()
                val temp03:Double = String.format("%.1f", temp02).toDouble()
                findViewById<TextView>(R.id.acc_X).text = temp03.toString()

                val accY = event.values[1]
                val temp11:Double = String.format("%.3f", accY).toDouble()
                val temp12:Double = String.format("%.2f", temp11).toDouble()
                val temp13:Double = String.format("%.1f", temp12).toDouble()
                findViewById<TextView>(R.id.acc_Y).text = temp13.toString()

                val accZ = event.values[2]
                val temp21:Double = String.format("%.3f", accZ).toDouble()
                val temp22:Double = String.format("%.2f", temp21).toDouble()
                val temp23:Double = String.format("%.1f", temp22).toDouble()
                findViewById<TextView>(R.id.acc_Z).text = temp23.toString()
            }

            if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
                val accX = event.values[0]
                val temp01:Double = String.format("%.3f", accX).toDouble()
                val temp02:Double = String.format("%.2f", temp01).toDouble()
                val temp03:Double = String.format("%.1f", temp02).toDouble()
                findViewById<TextView>(R.id.gyro_x).text = temp03.toString()

                val accY = event.values[1]
                val temp11:Double = String.format("%.3f", accY).toDouble()
                val temp12:Double = String.format("%.2f", temp11).toDouble()
                val temp13:Double = String.format("%.1f", temp12).toDouble()
                findViewById<TextView>(R.id.gyro_y).text = temp13.toString()

                val accZ = event.values[2]
                val temp21:Double = String.format("%.3f", accZ).toDouble()
                val temp22:Double = String.format("%.2f", temp21).toDouble()
                val temp23:Double = String.format("%.1f", temp22).toDouble()
                findViewById<TextView>(R.id.gyro_z).text = temp23.toString()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        mSensorManager.registerListener(this, mAccelerometer, 500000, 500000)
        mSensorManager.registerListener(this, mGyroscope, 500000, 500000)
    }

    override fun onPause() {
        super.onPause()
        mSensorManager.unregisterListener(this)
    }

    fun resumeReading(view: View) {
        this.resume = true
    }

    fun pauseReading(view: View) {
        this.resume = false
    }
}