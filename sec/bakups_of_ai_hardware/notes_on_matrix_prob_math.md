::wofl's curious attempt to progress ai design ii::

::overview::
aim is to boost llm capabilty by allowing it access to an analog vector/matrix/probability curve specialized computing unit.
- requires an interface between llm host computer and the analog computer.
- will likely have biggest challenges with noise, heat dissipation and clock issues.
- idea is interface will operate fast enough to let not just gain advantage from fast vector/matrix math but also significant enough speed boost to allow not only improved performance but even feedback loops/recursion at exceptional performance levels and perhaps even resonnce induced oscillations resulting in emergent behaviours in the system.


::adcdac specific considerations::
- cant get 128i/o's cheap n sm0l but can easily daisychain 16's/32's
- heat dissipation and noise as ever are our enemies
- ti/maxim etc. gud for it - see detailed convo for info 
not too many major considerations in this section as long as basics covered and pcb design is good for noise/heat dissipation
is worth thinkin about any issues yet covered arising from smd choice - adcdac package might affect package choice for others ofc.

::power considerations
will need
- power conditioing pre-/post- mains? i.e. filtering via LC or ferrites,or both
- initial switching down to +/-12V then further switchings for +3.3V digital/linears for +5V analog.
star grounding/multilayer boards with apt groundplanes
- ofc, as petey repeatedly nags, mustnt forget decouplings blahblah :D


::clock/comms/signal/etc. considerations::
with ascdac in common system, keeping all clock systems synchronous is essential.
main troublemaker is nouise - jitter, crosstalk, interference both internal/external:
- spi for main high speed comms, i2c for control config messages, status reports, and error reporting, gpio for signals like power resets etc.? these will require buffering everywhere.
- employ Impedance Matching and Differential Signaling everywhere
- likely need dma or interrupt implementation to cope with speeds required for feedback loops
- latency may require needs to be bournein mind at all times too since synchronicity and speed are of issue


::other considerations::
note: look into hotswap capability if gonna be a scale up thing?


::n.b. do we need some kind of shielded case beyond usual computer gear grounded metal/plastic?
likewise, are the separations (i.e.airgaps) between adcdac module boards sufficient to avoid intermodule noise issues [likely], or do need to sammich with blank pcbs that have ground plane?  m