package com.example.sensorapp30

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
import android.hardware.SensorManager.SENSOR_DELAY_NORMAL
import android.location.Location
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.widget.Button
import androidx.annotation.NonNull
import com.google.firebase.database.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.concurrent.scheduleAtFixedRate

class MainActivity : AppCompatActivity() {
    private lateinit var sensorManager: SensorManager
    private lateinit var accelerometer: Sensor
    private lateinit var gyroscope: Sensor
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    var count = 0
    var latitude : Double = 0.0
    var longitude : Double = 0.0

    // start button flag
    private var resume = false
    private var sendingData = false

    // category button click counts
    private var pothole_btn_click = 0
    private var speedbreaker_btn_click = 0
    private var traffic_btn_click = 0
    private var schange_btn_click = 0
    private var broad_btn_click = 0

    // category labels
    private var label = "Normal Road"
    private var pothole_l = "Pothole"
    private var speedbreaker_l = "Speedbreaker"
    private var traffic_l = "Traffic"
    private var schange_l = "Sudden Change"
    private var broad_l = "Bad Road"

    private var android_id : String = ""
    private var session_id : String = ""

    private var myList = mutableListOf<MutableList<String>>()

    // database initialization
    private lateinit var database: DatabaseReference

    var sensor_label = ""
    var accelerometer_x = "0.0"
    var accelerometer_y = "0.0"
    var accelerometer_z = "0.0"

    var gyroscope_x = "0.0"
    var gyroscope_y = "0.0"
    var gyroscope_z = "0.0"

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestWindowFeature(Window.FEATURE_NO_TITLE)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN)
        setContentView(R.layout.activity_main)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        database = FirebaseDatabase.getInstance().getReference("sensor-data2")

        android_id = Settings.Secure.getString(this.contentResolver, Settings.Secure.ANDROID_ID)

        session_id = UUID.randomUUID().toString()

        // ask for permission
        if(CheckPermission()){
            if(isLocationEnabled()){

                sensorManager.registerListener(
                    sensorEventListener, accelerometer, 20_000
                )

                sensorManager.registerListener(
                    sensorEventListener, gyroscope, 20_000
                )

                val locationRequest = LocationRequest.create().apply {
                    priority = LocationRequest.PRIORITY_HIGH_ACCURACY
                    interval = 500
                    fastestInterval = 500
                }
                fusedLocationClient.requestLocationUpdates(
                    locationRequest,
                    locationCallback,
                    Looper.getMainLooper()
                )

            }else{
                Toast.makeText(this, "Location service not enabled", Toast.LENGTH_SHORT).show()
            }

        }else{
            RequestPermission()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onResume()
    {
        super.onResume()
        sensorManager.registerListener(
            sensorEventListener, accelerometer, 1_000_000
        )
        sensorManager.registerListener(
            sensorEventListener, gyroscope, 100000
        )
        val locationRequest = LocationRequest.create().apply {
            priority = LocationRequest.PRIORITY_HIGH_ACCURACY
            interval = 1000
            fastestInterval = 1000
        }
        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            Looper.getMainLooper()
        )

        // check for internet connection after every 5 seconds
        val connection_timer = Timer()
        connection_timer.scheduleAtFixedRate(object: TimerTask()
        {
            override fun run()
            {
                if (!resume && myList.size > 20 && checkforInternet(this@MainActivity))
                {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Sending Data!", Toast.LENGTH_SHORT).show()
                    }
                    for (x in myList.drop(20))
                    {
                        sendingData = true
                        // add in database
                        database.child(session_id).child(x[0]).child("timestamp").setValue(x[0])
                        database.child(session_id).child(x[0]).child("android-id").setValue(x[1])
                        database.child(session_id).child(x[0]).child("label").setValue(x[2])
                        database.child(session_id).child(x[0]).child("latitude").setValue(x[3])
                        database.child(session_id).child(x[0]).child("longitude").setValue(x[4])
                        database.child(session_id).child(x[0]).child("accelerometer-x").setValue(x[5])
                        database.child(session_id).child(x[0]).child("accelerometer-y").setValue(x[6])
                        database.child(session_id).child(x[0]).child("accelerometer-z").setValue(x[7])
                        database.child(session_id).child(x[0]).child("gyroscope-x").setValue(x[8])
                        database.child(session_id).child(x[0]).child("gyroscope-y").setValue(x[9])
                        database.child(session_id).child(x[0]).child("gyroscope-z").setValue(x[10])

                        // check if last element in list is in database
                        if (myList.size == 21)
                        {
                            database.addValueEventListener(object : ValueEventListener {
                                override fun onDataChange(dataSnapshot : DataSnapshot)
                                {
                                    if (dataSnapshot.child(session_id).child(x[0]).exists())
                                    {
                                        myList.remove(x)
                                    }
                                }
                                override fun onCancelled(error : DatabaseError) {}
                            })
                        }
                        else
                        {
                            myList.remove(x)
                        }
                    }
                }
                else if (!resume && myList.size <= 20 && checkforInternet(this@MainActivity) && sendingData)
                {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Data Sent!", Toast.LENGTH_SHORT).show()
                    }
                    sendingData = false
                }
            }
        },5000, 5000)
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(sensorEventListener)
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }

    private val sensorEventListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent?) {
            if (event != null && resume)
            {
                println("///////////////////////////////")
                println(count)

                if (event != null && resume)
                {
                    val date = Date()
                    val formatting = SimpleDateFormat("yyyyMMddHHmmss-SSS")
                    val formatteddate = formatting.format(date)

                    val pothole_btn = findViewById<Button>(R.id.pothole)
                    val speedbreaker_btn = findViewById<Button>(R.id.speedbreaker)
                    val traffic_btn = findViewById<Button>(R.id.traffic)
                    val bad_road_btn = findViewById<Button>(R.id.badroad)
                    val sudden_change_btn = findViewById<Button>(R.id.suddenchange)

                    pothole_btn.setOnClickListener(object : View.OnClickListener {
                        override fun onClick(view: View)
                        {
                            pothole_btn_click = pothole_btn_click + 1
                            if (pothole_btn_click == 1)
                            {
                                sensor_label = pothole_l

                                speedbreaker_btn.setEnabled(false)
                                traffic_btn.setEnabled(false)
                                bad_road_btn.setEnabled(false)
                                sudden_change_btn.setEnabled(false)
                            }
                            else
                            {
                                sensor_label = label

                                pothole_btn_click = 0
                                speedbreaker_btn.setEnabled(true)
                                traffic_btn.setEnabled(true)
                                bad_road_btn.setEnabled(true)
                                sudden_change_btn.setEnabled(true)
                            }
                        }
                    })

                    speedbreaker_btn.setOnClickListener(object : View.OnClickListener
                    {
                        override fun onClick(view : View)
                        {
                            speedbreaker_btn_click = speedbreaker_btn_click + 1
                            if (speedbreaker_btn_click == 1)
                            {
                                sensor_label = speedbreaker_l
                                pothole_btn.setEnabled(false)
                                traffic_btn.setEnabled(false)
                                bad_road_btn.setEnabled(false)
                                sudden_change_btn.setEnabled(false)
                            }
                            else
                            {
                                sensor_label = label
                                speedbreaker_btn_click = 0
                                pothole_btn.setEnabled(true)
                                traffic_btn.setEnabled(true)
                                bad_road_btn.setEnabled(true)
                                sudden_change_btn.setEnabled(true)
                            }
                        }
                    })

                    traffic_btn.setOnClickListener(object : View.OnClickListener
                    {
                        override fun onClick(view : View)
                        {
                            traffic_btn_click = traffic_btn_click + 1
                            if (traffic_btn_click == 1)
                            {
                                sensor_label = traffic_l
                                pothole_btn.setEnabled(false)
                                speedbreaker_btn.setEnabled(false)
                                bad_road_btn.setEnabled(false)
                                sudden_change_btn.setEnabled(false)
                            }
                            else
                            {
                                sensor_label = label
                                traffic_btn_click = 0
                                pothole_btn.setEnabled(true)
                                speedbreaker_btn.setEnabled(true)
                                bad_road_btn.setEnabled(true)
                                sudden_change_btn.setEnabled(true)
                            }
                        }
                    })

                    bad_road_btn.setOnClickListener(object : View.OnClickListener
                    {
                        override fun onClick(view : View)
                        {
                            broad_btn_click = broad_btn_click + 1
                            if (broad_btn_click == 1)
                            {
                                sensor_label = broad_l
                                pothole_btn.setEnabled(false)
                                traffic_btn.setEnabled(false)
                                speedbreaker_btn.setEnabled(false)
                                sudden_change_btn.setEnabled(false)
                            }
                            else
                            {
                                sensor_label = label
                                broad_btn_click = 0
                                pothole_btn.setEnabled(true)
                                traffic_btn.setEnabled(true)
                                speedbreaker_btn.setEnabled(true)
                                sudden_change_btn.setEnabled(true)
                            }
                        }
                    })

                    sudden_change_btn.setOnClickListener(object : View.OnClickListener
                    {
                        override fun onClick(view : View)
                        {
                            schange_btn_click = schange_btn_click + 1
                            if (schange_btn_click == 1)
                            {
                                sensor_label = schange_l
                                pothole_btn.setEnabled(false)
                                traffic_btn.setEnabled(false)
                                bad_road_btn.setEnabled(false)
                                speedbreaker_btn.setEnabled(false)
                            }
                            else
                            {
                                sensor_label = label
                                schange_btn_click = 0
                                pothole_btn.setEnabled(true)
                                traffic_btn.setEnabled(true)
                                bad_road_btn.setEnabled(true)
                                speedbreaker_btn.setEnabled(true)
                            }
                        }
                    })

                    if (((pothole_btn_click == 1) or (traffic_btn_click == 1) or (speedbreaker_btn_click == 1) or (schange_btn_click == 1) or (broad_btn_click == 1)))
                    {
                        if (pothole_btn_click == 1)
                        {
                            sensor_label = pothole_l
                        }
                        else if (traffic_btn_click == 1)
                        {
                            sensor_label = traffic_l
                        }
                        else if (speedbreaker_btn_click == 1)
                        {
                            sensor_label = speedbreaker_l
                        }
                        else if (schange_btn_click == 1)
                        {
                            sensor_label = schange_l
                        }
                        else if (broad_btn_click == 1)
                        {
                            sensor_label = broad_l
                        }
                    }
                    else
                    {
                        sensor_label = label
                    }

                    try{
                        findViewById<TextView>(R.id.longt).text = longitude.toString()
                        findViewById<TextView>(R.id.lat).text = latitude.toString()
                    }
                    catch(e: Exception){
                        println("failed")
                    }

                    if (event.sensor?.type == Sensor.TYPE_ACCELEROMETER)
                    {
                        count = count + 1
                        if (count % 2 == 0)
                        {
                            // Process accelerometer data
                            val accX = event.values[0]
                            findViewById<TextView>(R.id.acc_X).text = accX.toString()

                            val accY = event.values[1]
                            findViewById<TextView>(R.id.acc_Y).text = accY.toString()

                            val accZ = event.values[2]
                            findViewById<TextView>(R.id.acc_Z).text = accZ.toString()

                            accelerometer_x = accX.toString()
                            accelerometer_y = accY.toString()
                            accelerometer_z = accZ.toString()

                            myList.add(mutableListOf(formatteddate, android_id, sensor_label, latitude.toString(), longitude.toString(), accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z))

                        }
                    }

                    if (event.sensor?.type == Sensor.TYPE_GYROSCOPE)
                    {
                        // Process gyroscope data
                        val gyrX = event.values[0]
                        findViewById<TextView>(R.id.gyro_x).text = gyrX.toString()

                        val gyrY = event.values[1]
                        findViewById<TextView>(R.id.gyro_y).text = gyrY.toString()

                        val gyrZ = event.values[2]
                        findViewById<TextView>(R.id.gyro_z).text = gyrZ.toString()

                        gyroscope_x = gyrX.toString()
                        gyroscope_y = gyrY.toString()
                        gyroscope_z = gyrZ.toString()
                    }

//                    if (count % 2 == 0)
//                    {
//                        myList.add(mutableListOf(formatteddate, android_id, sensor_label, latitude.toString(), longitude.toString(), accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z))
//                    }
                }
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }
    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(locationResult: LocationResult?) {
            locationResult ?: return
            for (location in locationResult.locations) {
                // Process location data
                latitude = location.latitude
                longitude = location.longitude
            }
        }
    }

    private fun CheckPermission() : Boolean {
        if( (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
            || (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) ){
            return true
        }

        return false
    }

    private fun RequestPermission() {

        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ), 1000
        )
    }

    private fun isLocationEnabled() : Boolean{
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
        }
    }

    private fun checkforInternet(context : Context) : Boolean
    {
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val network = connectivityManager.activeNetwork ?: return false
            val activeNetwork = connectivityManager.getNetworkCapabilities(network) ?: return false
            return when {
                activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> true
                activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> true
                else -> false
            }
        }
        else {
            @Suppress("DEPRECATION") val networkInfo = connectivityManager.activeNetworkInfo ?: return false
            @Suppress("DEPRECATION")
            return networkInfo.isConnected
        }
    }
}
