package com.example.application

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
import java.text.SimpleDateFormat
import java.util.*
import android.provider.Settings
import android.annotation.SuppressLint
import android.location.Location
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.widget.Button
import com.google.firebase.database.*
import kotlinx.coroutines.delay
import kotlin.concurrent.schedule

class MainActivity : AppCompatActivity(){

    // location initialization
    lateinit var fusedLocationProviderClient: FusedLocationProviderClient
    lateinit var locationRequest : LocationRequest
    lateinit var locationResult : Location
    var latitude : Double = 0.0
    var longitude : Double = 0.0

    // start button flag
    private var resume = false

    var x : String = ""
    var y : String = ""
    var z : String = ""


    // database initialization
    private lateinit var database: DatabaseReference

    private val sensorManager by lazy { getSystemService(Context.SENSOR_SERVICE) as SensorManager }
    private val accelerometerSensor by lazy { sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)}

    private val sensorEventListener = object : SensorEventListener {
        @SuppressLint("MissingPermission")

        override fun onSensorChanged(event: SensorEvent?)
        {
            if (event?.sensor?.type == Sensor.TYPE_ACCELEROMETER && resume)
            {
                // Handle accelerometer data
                x = event.values[0].toString()
                y = event.values[1].toString()
                z = event.values[2].toString()
                // update UI or perform other operations
                findViewById<TextView>(R.id.acc_X).text = x
                findViewById<TextView>(R.id.acc_Y).text = y
                findViewById<TextView>(R.id.acc_Z).text = z

                // get location information
                latitude = locationResult.latitude
                longitude = locationResult.longitude
                findViewById<TextView>(R.id.longt).text = longitude.toString()
                findViewById<TextView>(R.id.lat).text = latitude.toString()
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {

        fusedLocationProviderClient = LocationServices.getFusedLocationProviderClient(this)

        requestWindowFeature(Window.FEATURE_NO_TITLE)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN)

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        database = FirebaseDatabase.getInstance().getReference("sensor-data")
    }

    override fun onStart()
    {
        super.onStart()
        sensorManager.registerListener(sensorEventListener, accelerometerSensor, SensorManager.SENSOR_DELAY_NORMAL, 500000 )

        //start location updates
        getLastLocation()
    }

    override fun onResume() {
        super.onResume()

        // register sensor listener at a sampling rate of 5 Hz
        sensorManager.registerListener(sensorEventListener, accelerometerSensor, SensorManager.SENSOR_DELAY_NORMAL, 500000 )

        //start location updates
//        getLastLocation()
    }

    override fun onPause() {
        super.onPause()
        // unregister sensor listener and stop location updates
        sensorManager.unregisterListener(sensorEventListener)
        fusedLocationProviderClient.removeLocationUpdates(locationCallback)
    }

    // button start and stop
    fun resumeReading(view: View) {
        if (this.resume == true)
        {
            Toast.makeText(this, "Data collection has already been started!", Toast.LENGTH_SHORT).show()
        }
        else if (this.resume == false)
        {
            this.resume = true
            Toast.makeText(this, "Data collection has begun!", Toast.LENGTH_SHORT).show()

            // get location updates after every 1 second
            Timer().scheduleAtFixedRate( object : TimerTask() {
                override fun run() {
                    getLastLocation()
                }
            }, 0, 1000)

        }
    }

    fun pauseReading(view: View) {
        if (this.resume == false)
        {
            Toast.makeText(this, "Data collection has already stopped!", Toast.LENGTH_SHORT).show()
        }
        else if (this.resume == true)
        {
            this.resume = false
            Toast.makeText(this, "Data collection has stopped!", Toast.LENGTH_SHORT).show()
            fusedLocationProviderClient.removeLocationUpdates(locationCallback)
        }
    }

    // LOCATION STUFF
    @SuppressLint("MissingPermission")
    private fun getLastLocation() {
        if(CheckPermission()){
            if(isLocationEnabled())
            {
                fusedLocationProviderClient.lastLocation.addOnCompleteListener{task -> getNewLocation()}
            }
            else
            {
                Toast.makeText(this, "Location service not enabled", Toast.LENGTH_SHORT).show()
            }
        }
        else
        {
            RequestPermission()
        }
    }

    private fun CheckPermission() : Boolean
    {
        if( (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
            || (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) ){
            return true
        }
        return false
    }

    private fun RequestPermission()
    {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ), 1000
        )
    }

    private fun isLocationEnabled() : Boolean
    {
        val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        return locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) || locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == 1000){
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                println("Permission granted")
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun getNewLocation()
    {
        // request for location updates after every 1 second (1000 milliseconds)
        locationRequest = LocationRequest.create().apply {
            interval = 1000
            fastestInterval = 1000
            priority = LocationRequest.PRIORITY_HIGH_ACCURACY
        }
        fusedLocationProviderClient.requestLocationUpdates(locationRequest, locationCallback, null)
    }

    private val locationCallback = object : LocationCallback()
    {
        // get the new location
        override fun onLocationResult(p0: LocationResult?)
        {
            val lastLocation = p0?.lastLocation
            if (lastLocation != null)
            {
                locationResult = lastLocation
            }
        }
    }
}