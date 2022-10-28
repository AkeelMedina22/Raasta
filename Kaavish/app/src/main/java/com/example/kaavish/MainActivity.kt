package com.example.kaavish

import android.Manifest
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.*
import com.google.firebase.database.DatabaseReference
import com.google.firebase.database.FirebaseDatabase
import java.security.AccessController.getContext
import java.text.SimpleDateFormat
import java.util.*
import android.provider.Settings

class MainActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var mSensorManager : SensorManager
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var mAccelerometer : Sensor ?= null
    private var mGyroscope : Sensor ?= null
    private var resume = false
    // globally declare LocationRequest
    private lateinit var locationRequest: LocationRequest

    // globally declare LocationCallback
    private lateinit var locationCallback: LocationCallback

    // added this line - abeer
    //private val android_id = Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)
    private lateinit var database: DatabaseReference


    override fun onCreate(savedInstanceState: Bundle?) {


        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        requestWindowFeature(Window.FEATURE_NO_TITLE)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN)

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mSensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        database = FirebaseDatabase.getInstance().getReference("sensor-data")

        getLocationUpdates()

    }

    private fun getLocationUpdates()
    {

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        locationRequest = LocationRequest()
        locationRequest.interval = 50000
        locationRequest.fastestInterval = 50000
        locationRequest.smallestDisplacement = 170f // 170 m = 0.1 mile
        locationRequest.priority = LocationRequest.PRIORITY_HIGH_ACCURACY //set according to your app function

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(locationResult: LocationResult?) {
                println("hello")
                locationResult ?: return

                if (locationResult.locations.isNotEmpty()) {
                    // get latest location
                    val location =
                        locationResult.lastLocation
                    // use your location object
                    // get latitude , longitude and other info from this
                    val lat = location.latitude
                    findViewById<TextView>(R.id.lat).text = lat.toString()
                    val longt = location.longitude
                    findViewById<TextView>(R.id.longt).text = longt.toString()

                }
            }
        }
    }

    //start location updates
    private fun startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            null /* Looper */
        )
    }

    // stop location updates
    private fun stopLocationUpdates() {
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }


    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        print("accuracy changed")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        // update onlocation result here,, but how ?
//
//        // add timestamp to database record
//        database.child(uid).setValue(formatteddate)
//        database.child(uid).setValue((android_id))


        if (event != null && resume) {



            val date = Date()
            val formatting = SimpleDateFormat("yyyymmddhhmmss")
            val formatteddate = formatting.format(date)

//        val uid = android_id + "-" + formatteddate
//        println("uid")

            //add this to database
            database.child(formatteddate).child("timestamp").setValue(formatteddate).addOnSuccessListener {
                Toast.makeText(this, "Succesfully saved!", Toast.LENGTH_SHORT).show()
            }.addOnFailureListener{
                Toast.makeText(this, "Failed!", Toast.LENGTH_SHORT).show()
            }

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

                database.child(formatteddate).child("accelerometer-x").setValue(temp03.toString())
                database.child(formatteddate).child("accelerometer-y").setValue(temp13.toString())
                database.child(formatteddate).child("accelerometer-z").setValue(temp23.toString())

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

                database.child(formatteddate).child("gyroscope-x").setValue(temp03.toString())
                database.child(formatteddate).child("gyroscope-y").setValue(temp13.toString())
                database.child(formatteddate).child("gyroscope-z").setValue(temp23.toString())
            }
        }
    }

    override fun onResume() {
        super.onResume()
        mSensorManager.registerListener(this, mAccelerometer, 500000, 500000)
        mSensorManager.registerListener(this, mGyroscope, 500000, 500000)
        startLocationUpdates()
    }

    override fun onPause() {
        super.onPause()
        stopLocationUpdates()
        mSensorManager.unregisterListener(this)
    }

    fun resumeReading(view: View) {
        this.resume = true
    }

    fun pauseReading(view: View) {
        this.resume = false
    }
}