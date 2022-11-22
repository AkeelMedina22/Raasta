package com.example.raasta


import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.pm.PackageManager.PERMISSION_GRANTED
import android.location.Location
import android.location.LocationManager
import android.os.Bundle
import android.os.Looper
import android.view.Window
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.preference.PreferenceManager
import com.google.android.gms.location.*
import com.google.android.gms.tasks.CancellationToken
import com.google.android.gms.tasks.CancellationTokenSource
import com.google.android.gms.tasks.OnTokenCanceledListener
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.osmdroid.api.IMapController
import org.osmdroid.bonuspack.routing.OSRMRoadManager
import org.osmdroid.bonuspack.routing.RoadManager
import org.osmdroid.config.Configuration
import org.osmdroid.config.Configuration.getInstance
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.mylocation.MyLocationNewOverlay


class MainActivity : AppCompatActivity(){

    // location initialization
    val PERMISSION_ID = 42
    companion object {
        private const val UPDATE_INTERVAL_IN_MILLISECONDS = 500L
        private const val FASTEST_UPDATE_INTERVAL_IN_MILLISECONDS =
            UPDATE_INTERVAL_IN_MILLISECONDS / 2
    }
    lateinit var fusedLocationProviderClient: FusedLocationProviderClient
    lateinit var locationRequest : LocationRequest
    lateinit var locationResult : Location
    private lateinit var locationCallback: LocationCallback

    var latitude : Double = 0.0
    var longitude : Double = 0.0


    var start : Location? = null

    val roadManager: RoadManager = OSRMRoadManager(this, "Raasta")
    val waypoints = ArrayList<GeoPoint>()


    // map intialization
    private lateinit var map: MapView
    private lateinit var mapController: IMapController
    lateinit var myLocationOverlay: MyLocationNewOverlay

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        locationResult = Location("current_location")
        // layout initialization
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        // Request Location permission
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PERMISSION_GRANTED
        ) {
            println("Location Permission GRANTED")
        } else {
            println("Location Permission DENIED")
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                1
            )
        }

        // MAP
        Configuration.getInstance().setUserAgentValue(this.getPackageName())
        getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this))

        setContentView(R.layout.activity_main)

        map = findViewById<MapView>(R.id.map)
        map.setTileSource(TileSourceFactory.MAPNIK)


        // get current location
        fusedLocationProviderClient = LocationServices.getFusedLocationProviderClient(this)
        GlobalScope.launch{
            locationResult = getCurrentLocation()
        }

        println(locationResult.latitude)

        val mapController = map.controller
        val startPoint = GeoPoint(locationResult.latitude, locationResult.longitude)
        mapController.setZoom(18.5)
        mapController.setCenter(startPoint)


//        map.overlays.add(myLocationOverlay)

//        map.invalidate()

        //Offline maps:
        //map.setUseDataConnection(true);

        map.isClickable = true
        map.setMultiTouchControls(true)

//        val endPoint = GeoPoint(29.0, 70.0)
//        waypoints.add(endPoint)
//        val road : Road = roadManager.getRoad(waypoints)
//        val roadOverlay: Polyline = RoadManager.buildRoadOverlay(road)
//        map.getOverlays().add(roadOverlay)
//        map.invalidate()



    }

    override fun onResume()
    {
        super.onResume()
        map.onResume()
    }

    override fun onPause()
    {
        super.onPause()
        map.onPause()

    }

    suspend fun getCurrentLocation() : Location
    {
        val def = CompletableDeferred<Location>()
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PERMISSION_GRANTED)
        {
            fusedLocationProviderClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, object : CancellationToken() {
                override fun onCanceledRequested(listener: OnTokenCanceledListener) = CancellationTokenSource().token
                override fun isCancellationRequested() = false }).addOnSuccessListener{
                    if (it != null)
                    {
                        def.complete(it)
                    }
                }
        }
        return def.await()

    }
//    private fun getCurrentLocation()
//    {
//        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PERMISSION_GRANTED)
//        {
//            fusedLocationProviderClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, object : CancellationToken() {
//                override fun onCanceledRequested(listener: OnTokenCanceledListener) = CancellationTokenSource().token
//                override fun isCancellationRequested() = false }).addOnSuccessListener{
//                    if (it != null)
//                    {
//                        showMap(it)
//                    }
//                }
//        }
//    }

    private fun showMap(location : Location)
    {
        println(location.longitude)
        println(location.latitude)
        val mapController = map.controller
        val startPoint = GeoPoint(location.latitude, location.longitude)
        mapController.setZoom(18.5)
        mapController.setCenter(startPoint)

        val startMarker = Marker(map)
        startMarker.position = startPoint
        startMarker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
        map.overlays.add(startMarker)


    }

}