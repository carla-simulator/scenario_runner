# F.A.Q.

Some of the most common issues regarding ScenarioRunner are listed hereunder. Some more can be found in the [GitHub issues][issuelink] for the project. In case any doubt is not listed here, take a look at the forum and feel free to post it.
[issuelink]: https://github.com/carla-simulator/scenario_runner/issues
<div class="build-buttons">
<p>
<a href="https://forum.carla.org/c/using-carla/scenario-runner" target="_blank" class="btn btn-neutral" title="Go to the CARLA forum">
CARLA forum</a>
</p>
</div>

---

## Runtime errors
<!-- ======================================================================= -->
  <details>
    <summary><h5 style="display:inline">
    <code>TypeError: 'instancemethod' object has no attribute 'getitem'</code> in the agent navigation
    </h5></summary>

This issue is most likely caused by an outdated version of the Python Networkx package.  

__1.__ Remove the current installation.
(e.g. <code>sudo apt-get remove python-networkx</code>) 

__2.__ Install a more current one.
 <code>pip install --user networkx==2.2</code>.

  </details>

<!-- ======================================================================= -->
  <details>
    <summary><h5 style="display:inline">
    <code>No more scenarios... Exiting</code> with no visible scenario
    </h5></summary>

The output should be similar to the following.

```sh
Preparing scenario: FollowLeadingVehicle_1
ScenarioManager: Running scenario FollowVehicle
Resetting ego-vehicle!
Failure!
Resetting ego-vehicle!
ERROR: failed to destroy actor 527 : unable to destroy actor: not found
No more scenarios .... Exiting
```

If nothing is happening, it is most likely due to the fact that there is no program controlling the ego vehicle. 

__1.__ Run the scenario again.

__2.__ Run a program to control the ego vehicle. For example, the `manual_control.py`.  

Something should be happening when the ego vehicle starts moving. 

  </details>


<!-- ======================================================================= -->
  <details>
    <summary><h5 style="display:inline">
    ScenarioRunner exits with error when using <code>--debug</code> commandline parameter
    </h5></summary>

The output should be similar to the following.
```sh
UnicodeEncodeError: 'ascii' codec can't encode character '\u2713' in position 58: ordinal not in range(128)
```

The environment variable is missing. 

__1.__ Set the environment variable in the terminal. Edit the `~/.bashrc` to avoid setting it everytime. 

```sh
PYTHONIOENCODING=utf-8
```

  </details>

---
